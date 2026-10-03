"""Single-GPU Hugging Face VLM runtime for the native veRL Worker backend.

The veRL driver transports examples and sampled token IDs. This module owns
image loading, Qwen-compatible multimodal encoding, generation, and causally
aligned forward passes inside the GPU Worker. It does not import TRL or GOLD.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...proxy_prompt import build_proxy_student_messages, build_proxy_teacher_messages
from ...proxy_protocol import ProxyExample


_SEQUENCE_INPUT_KEYS = ("mm_token_type_ids", "token_type_ids")
_NEUTRAL_GENERATION_KWARGS = {
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "force_words_ids": None,
    "sequence_bias": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_eos_token_id": None,
    "typical_p": 1.0,
    "min_p": None,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "penalty_alpha": None,
}


@dataclass(frozen=True)
class EncodedPrompt:
    """One unpadded, encoded image/text prompt resident on the Worker device.

    ``input_ids`` is a separate CPU copy for exact q-minus identity checks.
    ``sha256`` covers all processor tensors, including the image features.
    """

    inputs: dict[str, Any]
    input_ids: Any
    sha256: str


def _checked_config(config: Mapping[str, Any]) -> tuple[str, str | None, int, bool]:
    if not isinstance(config, Mapping):
        raise ValueError("VLMRuntime config must be a mapping")
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    revision = config.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("model_revision must be a nonempty string or null")
    if not Path(model_id).is_dir() and revision is None:
        raise ValueError("a non-local model requires a pinned model_revision")
    setting = config.get("setting")
    if not isinstance(setting, Mapping):
        raise ValueError("VLMRuntime config requires a setting mapping")
    max_pixels = setting.get("max_pixels")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("setting.max_pixels must be a positive integer")
    if setting.get("bf16") is not False or setting.get("fp16") is not False:
        raise ValueError("native veRL VLM runtime requires FP32")
    if setting.get("enable_thinking") is not False:
        raise ValueError("native veRL VLM runtime requires enable_thinking=false")
    checkpointing = setting.get("gradient_checkpointing")
    if type(checkpointing) is not bool:
        raise ValueError("setting.gradient_checkpointing must be boolean")
    return model_id, revision, max_pixels, checkpointing


def _disable_dropout(model: Any) -> None:
    """Turn off module and config dropout without changing any model weights."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    seen: set[int] = set()

    def visit(config: Any) -> None:
        if config is None or id(config) in seen:
            return
        seen.add(id(config))
        for name, value in vars(config).items():
            if "dropout" in name and isinstance(value, (float, int)) and not isinstance(value, bool):
                setattr(config, name, 0.0)
            elif name.endswith("_config") and hasattr(value, "__dict__"):
                visit(value)

    visit(getattr(model, "config", None))


def _tensor_digest(inputs: Mapping[str, Any]) -> str:
    """Hash the exact processed prompt, including image tensors and their shapes."""
    import torch

    digest = hashlib.sha256()
    for key in sorted(inputs):
        value = inputs[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"processor output {key!r} must be a tensor")
        packed = value.detach().to("cpu").contiguous()
        header = json.dumps(
            {"key": key, "dtype": str(packed.dtype), "shape": list(packed.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        bytes_ = packed.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(bytes_).to_bytes(8, "big"))
        digest.update(bytes_)
    return digest.hexdigest()


class VLMRuntime:
    """One FP32 Student and frozen Teacher on the Ray Worker's visible CUDA GPU."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        model_id, revision, max_pixels, checkpointing = _checked_config(config)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("VLMRuntime requires torch and transformers in the veRL uv environment") from error
        if not torch.cuda.is_available():
            raise RuntimeError("VLMRuntime requires a visible CUDA GPU")

        # This helper changes the processor/tokenizer template only; its module
        # has no TRL import at import time. The same template is used by q+/q-.
        from ...train_proxy import configure_generation_terminators, configure_nonthinking_template

        self.device = torch.device("cuda:0")
        load_kwargs: dict[str, Any] = {"local_files_only": Path(model_id).is_dir()}
        if revision is not None:
            load_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left", **load_kwargs)
        self.template_sha256 = configure_nonthinking_template(self.processor)
        image_processor = self.processor.image_processor
        size = dict(image_processor.size)
        if size.get("shortest_edge", 0) > max_pixels:
            raise ValueError("setting.max_pixels is below image processor shortest_edge")
        size["longest_edge"] = max_pixels
        image_processor.size = size
        self.tokenizer = self.processor.tokenizer

        def load_model() -> Any:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.float32, attn_implementation="eager", **load_kwargs,
            )
            _disable_dropout(model)
            return model.to(self.device)

        self.student = load_model()
        self.teacher = load_model()
        self.eos_token_ids = configure_generation_terminators(self.student, self.tokenizer)
        self.teacher.generation_config.eos_token_id = list(self.eos_token_ids)
        if checkpointing:
            self.student.gradient_checkpointing_enable()
        self.student.config.use_cache = False
        self.student.train()
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def _encode(self, messages: list[dict[str, Any]], image_path: str) -> EncodedPrompt:
        import torch
        from PIL import Image

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("processor returned an empty chat prompt")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        # Qwen3.5 expects one list of images per text sample.
        batch = self.processor(
            images=[[image]], text=[prompt], padding=True,
            padding_side="left", add_special_tokens=False, return_tensors="pt",
        )
        inputs = dict(batch)
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("VLM processor must return tensor-only model inputs")
        ids = inputs.get("input_ids")
        attention = inputs.get("attention_mask")
        if ids is None or attention is None or ids.ndim != 2 or attention.shape != ids.shape or ids.shape[0] != 1:
            raise RuntimeError("VLM processor returned invalid input_ids/attention_mask")
        # Strip any left padding so the causal logit index is always L-1.
        valid = attention[0].bool()
        active = valid.nonzero(as_tuple=True)[0]
        if active.numel() == 0 or not bool(valid[int(active[0]):].all()):
            raise RuntimeError("VLM prompt attention mask is empty or non-contiguous")
        first = int(active[0])
        if first:
            old_length = ids.shape[1]
            for key in ("input_ids", "attention_mask", *_SEQUENCE_INPUT_KEYS, "position_ids"):
                if key in inputs and inputs[key].ndim >= 2 and inputs[key].shape[-1] == old_length:
                    inputs[key] = inputs[key][..., first:]
        input_ids = inputs["input_ids"][0].detach().to("cpu", dtype=torch.long).clone()
        sha256 = _tensor_digest(inputs)
        return EncodedPrompt(
            inputs={key: value.to(self.device) for key, value in inputs.items()},
            input_ids=input_ids,
            sha256=sha256,
        )

    def encode_student(self, example: ProxyExample) -> EncodedPrompt:
        if not isinstance(example, ProxyExample):
            raise TypeError("encode_student requires a ProxyExample")
        messages = build_proxy_student_messages(example)
        return self._encode(messages, example.student_image)

    def encode_teacher(self, example: ProxyExample) -> EncodedPrompt:
        if not isinstance(example, ProxyExample):
            raise TypeError("encode_teacher requires a ProxyExample")
        messages = build_proxy_teacher_messages(
            example.teacher_question or example.question, example.teacher_evidence,
        )
        return self._encode(messages, example.teacher_image)

    def _sample_response(self, encoded: EncodedPrompt, parameters: Mapping[str, Any],
                         *, capture_logits: bool) -> Any:
        """Sample exact response IDs; optionally retain raw HF generation logits."""
        import torch

        if not isinstance(encoded, EncodedPrompt) or not isinstance(parameters, Mapping):
            raise TypeError("generate requires EncodedPrompt and parameter mapping")
        max_new_tokens = parameters.get("max_new_tokens")
        temperature = parameters.get("temperature")
        top_p = parameters.get("top_p")
        top_k = parameters.get("top_k", 0)
        seed = parameters.get("seed")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        was_training = self.student.training
        self.student.eval()
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
                torch.manual_seed(seed)
                result = self.student.generate(
                    **encoded.inputs,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=top_k,
                    eos_token_id=list(self.eos_token_ids),
                    pad_token_id=(self.tokenizer.pad_token_id
                                  if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id),
                    use_cache=True,
                    return_dict_in_generate=capture_logits,
                    output_logits=capture_logits,
                    **_NEUTRAL_GENERATION_KWARGS,
                )
        finally:
            self.student.train(was_training)
        sequences = getattr(result, "sequences", result)
        prompt_ids = encoded.inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]
        if sequences.ndim != 2 or sequences.shape[0] != 1 or sequences.shape[1] <= prompt_length:
            raise RuntimeError("VLM generate returned no response tokens")
        if not torch.equal(sequences[0, :prompt_length], prompt_ids[0]):
            raise RuntimeError("VLM generate changed the encoded prompt token IDs")
        response_ids = sequences[0, prompt_length:].detach().to("cpu", dtype=torch.long).clone()
        if not capture_logits:
            return response_ids
        raw_logits = getattr(result, "logits", None)
        if raw_logits is None or len(raw_logits) != response_ids.numel():
            raise RuntimeError("HF generation did not return one raw logit vector per sampled ID")
        stacked = torch.stack([step[0].detach().to("cpu", dtype=torch.float32) for step in raw_logits])
        if stacked.ndim != 2 or stacked.shape[0] != response_ids.numel():
            raise RuntimeError("HF raw generation logits do not align with sampled IDs")
        return response_ids, stacked

    def generate(self, encoded: EncodedPrompt, parameters: Mapping[str, Any]) -> Any:
        """Sample response IDs from the current Student; never decode/re-tokenize."""
        return self._sample_response(encoded, parameters, capture_logits=False)

    def generate_with_logits(self, encoded: EncodedPrompt, parameters: Mapping[str, Any]) -> Any:
        """Audit-only path: return the exact sampled IDs and raw generation logits."""
        return self._sample_response(encoded, parameters, capture_logits=True)

    def forward_response(self, model: Any, encoded: EncodedPrompt, completion_ids: Any) -> Any:
        """Return logits for each sampled token at position prompt_len+t-1.

        The caller chooses the gradient context: Student updates keep autograd,
        while Teacher and detached rollout scoring use ``torch.no_grad()``.
        """
        import torch

        if not isinstance(encoded, EncodedPrompt) or not isinstance(completion_ids, torch.Tensor):
            raise TypeError("forward_response requires EncodedPrompt and a tensor of token IDs")
        if completion_ids.ndim != 1 or completion_ids.dtype != torch.long or completion_ids.numel() == 0:
            raise ValueError("completion_ids must be a nonempty 1D LongTensor")
        if completion_ids.device.type != "cpu":
            raise ValueError("completion_ids must be on CPU for exact ID transport")
        inputs = dict(encoded.inputs)
        prompt_ids = inputs["input_ids"]
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
            raise RuntimeError("encoded prompt has invalid input_ids")
        if not torch.equal(prompt_ids[0].detach().to("cpu"), encoded.input_ids):
            raise RuntimeError("encoded prompt IDs changed since generation")
        response = completion_ids.to(self.device).unsqueeze(0)
        prompt_length = prompt_ids.shape[1]
        count = response.shape[1]
        inputs["input_ids"] = torch.cat((prompt_ids, response), dim=1)
        inputs["attention_mask"] = torch.cat((
            inputs["attention_mask"],
            inputs["attention_mask"].new_ones((1, count)),
        ), dim=1)
        for key in _SEQUENCE_INPUT_KEYS:
            if key in inputs:
                prefix = inputs[key]
                if prefix.ndim != 2 or prefix.shape != prompt_ids.shape:
                    raise RuntimeError(f"{key} does not align with prompt input_ids")
                inputs[key] = torch.cat((prefix, prefix.new_zeros((1, count))), dim=1)
        if "position_ids" in inputs:
            prefix = inputs["position_ids"]
            if prefix.ndim < 2 or prefix.shape[-1] != prompt_length:
                raise RuntimeError("position_ids do not align with prompt input_ids")
            offsets = torch.arange(1, count + 1, device=prefix.device, dtype=prefix.dtype)
            inputs["position_ids"] = torch.cat((prefix, prefix[..., -1:] + offsets), dim=-1)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < prompt_length + count - 1:
            raise RuntimeError("VLM forward returned too few causal logits")
        selected = logits[0, prompt_length - 1:prompt_length - 1 + count, :].clone()
        if selected.shape[0] != count:
            raise RuntimeError("VLM response logits do not align with completion IDs")
        return selected
