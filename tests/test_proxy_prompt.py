import unittest

from sure_vl.proxy_prompt import (
    build_proxy_masks, build_proxy_prompt, parse_proxy_completion, split_proxy_generated_eos,
)
from sure_vl.proxy_protocol import ProxyExample, verify_proxy_answer


def _example() -> ProxyExample:
    return ProxyExample.from_dict({
        "id": "one", "split": "dev", "student_image": "blur.png",
        "teacher_image": "clear.png", "question": "Which shape?",
        "accepted_answers": ["circle"],
    })


def _completion() -> str:
    return (
        "<vision>A red circle is visible.</vision>"
        "<answer>circle</answer>"
        "<confidence><visual_confidence>8</visual_confidence>"
        "<answer_confidence>7</answer_confidence></confidence>"
    )


class CharacterTokenizer:
    eos_token_id = 0

    def decode(self, ids, **_):
        return "".join(chr(value) for value in ids)

    def encode(self, text, **_):
        return [ord(character) for character in text]

    def __call__(self, text, **_):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces
        self.ids = list(range(1, len(pieces) + 1))

    def decode(self, ids, **_):
        return "".join(self.pieces[index - 1] for index in ids)

    def __call__(self, text, **_):
        if text != "".join(self.pieces):
            raise ValueError("unexpected text")
        offsets = []
        start = 0
        for piece in self.pieces:
            offsets.append((start, start + len(piece)))
            start += len(piece)
        return {"input_ids": self.ids, "offset_mapping": offsets}


class ProxyPromptTests(unittest.TestCase):
    def test_prompt_has_no_fact_slots_or_answer_leak(self) -> None:
        prompt = build_proxy_prompt(_example())
        self.assertIn("<vision>", prompt)
        self.assertIn("<answer_confidence>", prompt)
        self.assertIn("without reasoning", prompt)
        self.assertIn("at most 40 words", prompt)
        self.assertIn("output only the number", prompt)
        self.assertIn("one integer from 0 to 10", prompt)
        self.assertIn("internal certainty", prompt)
        self.assertIn("unconditional chance", prompt)
        self.assertTrue(prompt.endswith("Question: Which shape?"))
        self.assertLess(prompt.index("<vision>"), prompt.index("<answer>"))
        self.assertLess(prompt.index("<answer>"), prompt.index("<confidence>"))
        self.assertNotIn(" / ", prompt)
        self.assertNotIn("<think>", prompt)
        self.assertNotIn("<reasoning>", prompt)
        self.assertNotIn("required_visual_facts", prompt)
        self.assertNotIn("Your final answer", prompt)
        self.assertNotIn("0-100 integer", prompt)
        self.assertNotIn("brief free-text visual description", prompt)
        self.assertNotIn("circle", prompt.lower())
        example = prompt.split("Unrelated format example; do not copy its objects, answer, or scores:\n", 1)[1]
        example = example.split("Now answer using the actual image and question, with tags only.", 1)[0]
        parsed_example = parse_proxy_completion(_example(), example)
        self.assertTrue(parsed_example.format_valid)
        self.assertEqual(parsed_example.answer, "umbrella")
        self.assertEqual((parsed_example.visual_confidence, parsed_example.answer_confidence), (8, 7))

    def test_valid_output_and_exact_three_masks(self) -> None:
        text = _completion()
        parsed = parse_proxy_completion(_example(), text)
        self.assertTrue(parsed.format_valid)
        self.assertEqual(parsed.answer, "circle")
        self.assertEqual(parsed.visual_confidence, 8)
        self.assertEqual(parsed.answer_confidence, 7)
        self.assertIsNone(parsed.reasoning_text)
        self.assertTrue(verify_proxy_answer(_example(), parsed.answer))
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        vision_start, vision_end = parsed.vision_span_chars
        self.assertEqual(sum(masks.vision_mask), vision_end - vision_start)
        self.assertEqual(masks.vision_mask[vision_start:vision_end], (1,) * (vision_end - vision_start))
        self.assertEqual(sum(masks.content_mask), parsed.report_start_char)
        self.assertEqual(sum(masks.report_mask), len(text) - parsed.report_start_char)
        self.assertIsNone(masks.failure_reason)

    def test_scores_accept_only_exact_integers_from_zero_through_ten(self) -> None:
        base = _completion()
        for value in ("0", "9", "10"):
            text = base.replace("<visual_confidence>8</visual_confidence>",
                                f"<visual_confidence>{value}</visual_confidence>")
            parsed = parse_proxy_completion(_example(), text)
            self.assertTrue(parsed.format_valid, value)
            self.assertEqual(parsed.visual_confidence, int(value))
        for value in ("11", "80", "8.5", "08", "-1", " 8", "8 "):
            text = base.replace("<visual_confidence>8</visual_confidence>",
                                f"<visual_confidence>{value}</visual_confidence>")
            parsed = parse_proxy_completion(_example(), text)
            self.assertIsNone(parsed.visual_confidence, value)
            self.assertIn("invalid_visual_confidence", parsed.format_errors, value)
            self.assertEqual(parsed.answer_confidence, 7)
            tokenizer = CharacterTokenizer()
            masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
            self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))
        for value in ("11", "80", "7.5"):
            text = base.replace("<answer_confidence>7</answer_confidence>",
                                f"<answer_confidence>{value}</answer_confidence>")
            parsed = parse_proxy_completion(_example(), text)
            self.assertIsNone(parsed.answer_confidence, value)
            self.assertIn("invalid_answer_confidence", parsed.format_errors, value)
            self.assertEqual(parsed.visual_confidence, 8)

    def test_missing_vision_preserves_answer_and_teacher_content(self) -> None:
        text = (
            "<answer>circle</answer>"
            "<confidence><visual_confidence>0</visual_confidence>"
            "<answer_confidence>8</answer_confidence></confidence>"
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertIn("missing_or_invalid_vision", parsed.format_errors)
        self.assertIsNone(parsed.vision_span_chars)
        self.assertTrue(verify_proxy_answer(_example(), parsed.answer))
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.vision_mask), 0)
        self.assertGreater(sum(masks.content_mask), 0)
        self.assertGreater(sum(masks.report_mask), 0)

    def test_missing_confidence_keeps_every_token_for_teacher(self) -> None:
        text = _completion().split("<confidence>")[0]
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.answer, "circle")
        self.assertIn("missing_confidence", parsed.format_errors)
        self.assertIsNone(parsed.report_start_char)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.content_mask), len(text))
        self.assertEqual(sum(masks.report_mask), 0)
        self.assertGreater(sum(masks.vision_mask), 0)

    def test_partial_confidence_keeps_content_before_marker(self) -> None:
        text = _completion().split("<confidence>")[0] + "<confidence><visual_confidence>4</visual_confidence>"
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.visual_confidence, 4)
        self.assertIsNone(parsed.answer_confidence)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertGreater(sum(masks.content_mask), 0)
        self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))

    def test_literal_report_marker_inside_vision_does_not_cut_content(self) -> None:
        text = _completion().replace("A red circle is visible.", "A label says <confidence> beside a circle.")
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.report_start_char, text.rindex("<confidence>"))
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.content_mask), parsed.report_start_char)

    def test_misordered_report_marker_still_excludes_report_from_teacher_loss(self) -> None:
        text = (
            "<vision>A circle.</vision>"
            "<confidence><visual_confidence>6</visual_confidence>"
            "<answer_confidence>7</answer_confidence></confidence>"
            "<answer>circle</answer>"
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.answer, "circle")
        self.assertFalse(parsed.format_valid)
        self.assertEqual(parsed.report_start_char, text.index("<confidence>"))
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.content_mask), parsed.report_start_char)
        self.assertEqual(masks.report_mask[parsed.report_start_char:], (1,) * (len(text) - parsed.report_start_char))

    def test_literal_marker_inside_answer_is_not_report(self) -> None:
        text = _completion().split("<confidence>")[0].replace("<answer>circle</answer>", "<answer>a sign says <confidence></answer>")
        parsed = parse_proxy_completion(_example(), text)
        self.assertIsNone(parsed.report_start_char)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.content_mask), len(text))

    def test_vision_after_reasoning_is_not_eligible_for_proxy(self) -> None:
        text = (
            "<think><reasoning>It looks round.</reasoning><vision>A circle.</vision></think>"
            "<answer>circle</answer>"
            "<confidence><visual_confidence>5</visual_confidence>"
            "<answer_confidence>6</answer_confidence></confidence>"
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertIsNone(parsed.vision_span_chars)
        self.assertIn("missing_or_invalid_vision", parsed.format_errors)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.vision_mask), 0)
        self.assertGreater(sum(masks.content_mask), 0)

    def test_legacy_thinking_output_remains_readable_but_is_noncanonical(self) -> None:
        text = (
            "<think><vision>A red circle is visible.</vision>"
            "<reasoning>The shape is round.</reasoning></think>"
            "<answer>circle</answer><confidence>"
            "<visual_confidence>8</visual_confidence>"
            "<answer_confidence>7</answer_confidence></confidence>"
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.vision_text, "A red circle is visible.")
        self.assertEqual(parsed.reasoning_text, "The shape is round.")
        self.assertEqual(parsed.answer, "circle")
        self.assertIn("noncanonical_structure", parsed.format_errors)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertGreater(sum(masks.vision_mask), 0)
        self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))

    def test_direct_vision_after_answer_is_not_proxy_eligible(self) -> None:
        text = _completion().replace(
            "<vision>A red circle is visible.</vision><answer>circle</answer>",
            "<answer>circle</answer><vision>A red circle is visible.</vision>",
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertIsNone(parsed.vision_span_chars)
        self.assertIn("missing_or_invalid_vision", parsed.format_errors)
        self.assertEqual(parsed.answer, "circle")

    def test_cross_boundary_tokens_are_excluded_from_vision_and_report_content(self) -> None:
        text = _completion()
        pieces = [
            "<vision>A red", " circle is visible.", "</vision>",
            "<answer>circle", "</answer><confidence>",
            "<visual_confidence>8</visual_confidence><answer_confidence>7</answer_confidence></confidence>",
        ]
        self.assertEqual("".join(pieces), text)
        tokenizer = PieceTokenizer(pieces)
        parsed = parse_proxy_completion(_example(), text)
        masks = build_proxy_masks(tokenizer, tokenizer.ids, text, parsed)
        self.assertEqual(masks.vision_mask, (0, 1, 0, 0, 0, 0))
        self.assertEqual(masks.content_mask, (1, 1, 1, 1, 0, 0))
        self.assertEqual(masks.report_mask, (0, 0, 0, 0, 1, 1))

    def test_trailing_terminator_is_removed_without_trimming_body(self) -> None:
        tokenizer = CharacterTokenizer()
        body, terminal = split_proxy_generated_eos(tokenizer, [65, 66, 0])
        self.assertEqual((body, terminal), ((65, 66), (0,)))
        body, terminal = split_proxy_generated_eos(tokenizer, [65, 66, 99], generation_eos_token_id=99)
        self.assertEqual((body, terminal), ((65, 66), (99,)))


if __name__ == "__main__":
    unittest.main()
