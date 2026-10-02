"""Frozen-per-step teachers and evidence for successful optimizer updates."""

from __future__ import annotations

import math
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]


class EMATeacher:
    """Maintain FP32 EMA masters while using the teacher's forward dtype.

    The masters stay on the teacher's device. This avoids rounding away small
    successive updates when the forward model uses fp16/bf16. The caller must
    invoke ``update`` only after a successful student optimizer update.
    """

    def __init__(self, student: Any, teacher: Any, *, decay: float) -> None:
        import torch

        if isinstance(decay, bool) or not isinstance(decay, (float, int)) or not math.isfinite(decay):
            raise ValueError("EMA decay must be a finite number")
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.student = student
        self.teacher = teacher
        self.update_count = 0
        student_parameters = dict(student.named_parameters())
        self.teacher_parameters = dict(teacher.named_parameters())
        if student_parameters.keys() != self.teacher_parameters.keys():
            raise ValueError("student and EMA teacher parameter names differ")
        self.masters = {}
        for name, parameter in self.teacher_parameters.items():
            source = student_parameters[name]
            if source.shape != parameter.shape:
                raise ValueError(f"EMA parameter shape differs: {name}")
            if not parameter.is_floating_point():
                raise ValueError(f"EMA parameter must be floating point: {name}")
            parameter.requires_grad_(False)
            self.masters[name] = parameter.detach().to(dtype=torch.float32).clone()
        self.teacher.eval()

    def update(self) -> None:
        import torch

        student_parameters = dict(self.student.named_parameters())
        with torch.no_grad():
            for name, master in self.masters.items():
                source = student_parameters[name].detach().to(device=master.device, dtype=master.dtype)
                master.mul_(self.decay).add_(source, alpha=1 - self.decay)
                self.teacher_parameters[name].copy_(master)
            student_buffers = dict(self.student.named_buffers())
            for name, buffer in self.teacher.named_buffers():
                if name in student_buffers and buffer.shape == student_buffers[name].shape:
                    buffer.copy_(student_buffers[name].detach())
        self.teacher.eval()
        self.update_count += 1


class OptimizerEvidenceCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Count actual updates, and update EMA after successful optimizer steps."""

    def __init__(self, trainer: Any, *, teacher_mode: str, ema_decay: float) -> None:
        if teacher_mode not in {"fixed", "ema"}:
            raise ValueError("teacher_mode must be 'fixed' or 'ema'")
        self.trainer = trainer
        self.teacher_mode = teacher_mode
        self.ema_decay = ema_decay
        self.attempted_steps = 0
        self.successful_updates = 0
        self.skipped_updates = 0
        self.ema: EMATeacher | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if int(state.global_step) != 0:
            raise ValueError("proxy v1 does not support resuming EMA or optimizer evidence")
        if self.teacher_mode == "ema":
            accelerator = self.trainer.accelerator
            student = accelerator.unwrap_model(self.trainer.model)
            teacher = accelerator.unwrap_model(self.trainer.teacher_model)
            self.ema = EMATeacher(student, teacher, decay=self.ema_decay)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.attempted_steps += 1
        if self.trainer.accelerator.optimizer_step_was_skipped:
            self.skipped_updates += 1
        else:
            self.successful_updates += 1
            if self.ema is not None:
                self.ema.update()
        return control

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Reject nonfinite gradients before they can corrupt an FP32 model."""
        import torch

        gradients = [parameter.grad for parameter in self.trainer.model.parameters()
                     if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("optimizer update has no student gradients")
        norms = torch.stack([gradient.detach().norm() for gradient in gradients])
        if not bool(torch.isfinite(norms).all()):
            raise RuntimeError("nonfinite student gradients before optimizer update")
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> Any:
        if logs is not None:
            logs.update(self.summary())
        return control

    def summary(self) -> dict[str, Any]:
        return {
            "optimizer_attempted_steps": self.attempted_steps,
            "optimizer_successful_updates": self.successful_updates,
            "optimizer_skipped_updates": self.skipped_updates,
            "teacher_mode": self.teacher_mode,
            "teacher_ema_updates": self.ema.update_count if self.ema is not None else 0,
        }
