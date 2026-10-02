"""Tests for the VL-style text contract and exact content/report boundary."""

import unittest

from sure_vl.protocol import Example, ProtocolError, verify
from sure_vl.trl_prompt import (
    build_content_report_masks,
    build_user_prompt,
    parse_student_completion,
    split_generated_eos,
)


def example():
    return Example.from_dict(
        {
            "id": "sample-1",
            "split": "dev",
            "student_image": "restricted.png",
            "teacher_image": "clear.png",
            "question": "What is shown in the image?",
            "required_visual_facts": {"object": "private-square", "颜色": "private-blue"},
            "accepted_answers": ["private-square"],
        }
    )


def completion(vision='{"object":"private-square","颜色":"private-blue"}', visual=80, conditional=90):
    return (
        "<think><vision>" + vision + "</vision>"
        "<reasoning>The requested facts support the answer.</reasoning></think>\n"
        r"\boxed{private-square}" + "\n"
        "<confidence><vision_confidence>" + str(visual) + "</vision_confidence>"
        "<conditional_answer_confidence>" + str(conditional)
        + "</conditional_answer_confidence></confidence>"
    )


class PieceTokenizer:
    """A toy tokenizer whose pieces expose offsets and a possible straddle."""

    eos_token_id = 999

    def __init__(self, pieces):
        self.pieces = tuple(pieces)

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[token_id] for token_id in ids)

    def encode(self, text, *, add_special_tokens=False):
        if text == "".join(self.pieces):
            return list(range(len(self.pieces)))
        if text == self.pieces[0]:
            return [0]
        return []

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        if text != "".join(self.pieces):
            raise ValueError("toy tokenizer only knows its full string")
        offsets = []
        start = 0
        for piece in self.pieces:
            offsets.append((start, start + len(piece)))
            start += len(piece)
        return {"input_ids": list(range(len(self.pieces))), "offset_mapping": offsets}


class PromptTests(unittest.TestCase):
    def test_prompt_lists_slots_without_leaking_fact_values_or_reference_image(self):
        prompt = build_user_prompt(example())
        self.assertIn('<vision>{"object": "visible value", "颜色": "visible value"}</vision>', prompt)
        self.assertIn("<conditional_answer_confidence>", prompt)
        self.assertNotIn("private-square", prompt)
        self.assertNotIn("private-blue", prompt)
        self.assertNotIn("clear.png", prompt)
        self.assertNotIn("<analysis>", prompt)

    def test_parser_returns_student_output_and_exact_report_boundary(self):
        text = completion()
        parsed = parse_student_completion(example(), text)
        self.assertEqual(parsed.report_start_char, text.index("<confidence>"))
        self.assertEqual(parsed.output.id, "sample-1")
        self.assertEqual(parsed.output.visual_confidence, 80)
        self.assertEqual(parsed.output.conditional_answer_confidence, 90)
        self.assertTrue(verify(example(), parsed.output).visual_correct)
        self.assertTrue(verify(example(), parsed.output).answer_correct)

    def test_missing_fact_remains_parsed_and_scores_incorrect(self):
        parsed = parse_student_completion(example(), completion(vision='{"object":"private-square"}'))
        self.assertFalse(verify(example(), parsed.output).visual_correct)

    def test_strict_parser_rejects_unknown_duplicate_or_nonstring_facts(self):
        invalid_visions = (
            '{"object":"private-square","extra":"value"}',
            '{"object":"a","object":"b"}',
            '{"object":42}',
        )
        for vision in invalid_visions:
            with self.subTest(vision=vision), self.assertRaises(ProtocolError):
                parse_student_completion(example(), completion(vision=vision))

    def test_strict_parser_rejects_bad_report_and_extra_output(self):
        for text in (
            completion(visual=101),
            completion(visual="08"),
            completion(conditional="90.0"),
            completion() + " extra",
            completion().replace("<confidence>", "<analysis>maybe</analysis><confidence>"),
        ):
            with self.subTest(text=text[-80:]), self.assertRaises(ProtocolError):
                parse_student_completion(example(), text)

    def test_nested_math_braces_in_boxed_answer(self):
        text = completion().replace(r"\boxed{private-square}", r"\boxed{\frac{1}{2}}")
        parsed = parse_student_completion(example(), text)
        self.assertEqual(parsed.output.answer, r"\frac{1}{2}")

    def test_masks_split_exactly_at_confidence_token(self):
        text = completion()
        boundary = text.index("<confidence>")
        tokenizer = PieceTokenizer([text[:boundary], "<confidence>", text[boundary + len("<confidence>"):]])
        masks = build_content_report_masks(tokenizer, [0, 1, 2], text, boundary)
        self.assertEqual(masks.content_mask, (1, 0, 0))
        self.assertEqual(masks.report_mask, (0, 1, 1))
        self.assertEqual(masks.report_start_token, 1)
        self.assertIsNone(masks.failure_reason)

    def test_masks_assign_marker_straddle_to_report(self):
        text = completion()
        boundary = text.index("<confidence>")
        tokenizer = PieceTokenizer([text[:boundary - 1], text[boundary - 1:boundary + 3], text[boundary + 3:]])
        masks = build_content_report_masks(tokenizer, [0, 1, 2], text, boundary)
        self.assertEqual(masks.content_mask, (1, 0, 0))
        self.assertEqual(masks.report_mask, (0, 1, 1))
        self.assertEqual(masks.report_start_token, 1)
        self.assertFalse(masks.boundary_exact)
        self.assertIsNone(masks.failure_reason)

    def test_masks_fail_closed_for_malformed_or_mismatched_text(self):
        text = completion()
        boundary = text.index("<confidence>")
        tokenizer = PieceTokenizer([text[:boundary], text[boundary:]])
        for supplied_text, supplied_boundary in ((text, None), (text + "!", boundary), (text, boundary - 1)):
            with self.subTest(supplied_text=supplied_text[-8:], supplied_boundary=supplied_boundary):
                masks = build_content_report_masks(tokenizer, [0, 1], supplied_text, supplied_boundary)
                self.assertEqual(masks.content_mask, (0, 0))
                self.assertEqual(masks.report_mask, (1, 1))
                self.assertIsNotNone(masks.failure_reason)

    def test_split_generated_eos_marks_only_one_trailing_token(self):
        tokenizer = PieceTokenizer(["content", "report"])
        self.assertEqual(split_generated_eos(tokenizer, [0, 1, 999]), ((0, 1), (999,)))
        self.assertEqual(split_generated_eos(tokenizer, [0, 1]), ((0, 1), ()))
        self.assertEqual(
            split_generated_eos(tokenizer, [0, 1, 998], generation_eos_token_id=[999, 998]),
            ((0, 1), (998,)),
        )


if __name__ == "__main__":
    unittest.main()
