import unittest

from sure_vl.proxy_prompt import (
    PROXY_OUTPUT_PROTOCOL, build_proxy_masks, build_proxy_prompt, build_proxy_student_messages,
    build_proxy_teacher_messages,
    parse_proxy_completion, split_proxy_generated_eos,
)
from sure_vl.proxy_protocol import ProxyExample, ProxyProtocolError, verify_proxy_answer


def _example() -> ProxyExample:
    return ProxyExample.from_dict({
        "id": "one", "split": "dev", "student_image": "blur.png",
        "teacher_image": "clear.png", "question": "Which shape?",
        "accepted_answers": ["circle"],
    })


def _completion() -> str:
    return (
        "<vision>A red circle is visible.</vision>"
        "<reason>The visible shape is round.</reason>"
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
        self.assertEqual(PROXY_OUTPUT_PROTOCOL, "vision-reason-answer-confidence-v2")
        self.assertIn("<vision>", prompt)
        self.assertIn("<reason>", prompt)
        self.assertIn("<answer_confidence>", prompt)
        self.assertIn("<vision> within 40 words", prompt)
        self.assertIn("<reason> within 60 words", prompt)
        self.assertIn("output only the number", prompt)
        self.assertLess(prompt.index("output only the option letter"),
                        prompt.index("output only the number"))
        self.assertIn("two integer scores from 0 to 10", prompt)
        self.assertIn("internal certainty", prompt)
        self.assertIn("unconditional chance", prompt)
        self.assertTrue(prompt.endswith("Question: Which shape?"))
        self.assertLess(prompt.index("<vision>"), prompt.index("<answer>"))
        self.assertLess(prompt.index("<vision>"), prompt.index("<reason>"))
        self.assertLess(prompt.index("<reason>"), prompt.index("<answer>"))
        self.assertLess(prompt.index("<answer>"), prompt.index("<confidence>"))
        self.assertNotIn(" / ", prompt)
        self.assertNotIn("<think>", prompt)
        self.assertNotIn("<reasoning>", prompt)
        self.assertNotIn("required_visual_facts", prompt)
        self.assertNotIn("Your final answer", prompt)
        self.assertNotIn("0-100 integer", prompt)
        self.assertNotIn("brief free-text visual description", prompt)
        self.assertNotIn("circle", prompt.lower())
        self.assertIn("You FIRST identify the question-relevant visual evidence", prompt)
        self.assertIn("<vision>...</vision>", prompt)
        self.assertIn("<reason>...</reason>", prompt)
        self.assertNotIn("<answer>B</answer>", prompt)
        self.assertNotIn("<visual_confidence>8</visual_confidence>", prompt)

    def test_student_hint_precedes_question_and_teacher_evidence_is_absent(self) -> None:
        raw = _example().to_dict()
        raw["student_image_hint"] = "Only focus on the region inside the red bounding box."
        raw["teacher_evidence"] = {"secret_evidence_marker": "private scene graph"}
        prompt = build_proxy_prompt(ProxyExample.from_dict(raw))
        self.assertIn(raw["student_image_hint"], prompt)
        self.assertLess(prompt.index(raw["student_image_hint"]), prompt.index("Question: Which shape?"))
        self.assertTrue(prompt.endswith("Question: Which shape?"))
        self.assertNotIn("secret_evidence_marker", prompt)
        self.assertNotIn("private scene graph", prompt)

    def test_teacher_messages_are_distinct_and_privileged_evidence_is_scoped(self) -> None:
        evidence = {"scene_graph": [{"shape": "rectangle", "id": "evidence-1"}]}
        messages = build_proxy_teacher_messages("Which shape?", evidence)
        self.assertEqual(len(messages), 2)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"][0], {"type": "image"})
        teacher_system = messages[0]["content"][0]["text"]
        teacher_user = messages[1]["content"][1]["text"]
        self.assertIn("visual teacher", teacher_system)
        self.assertIn("enhanced question-relevant regional view", teacher_system)
        self.assertIn("target area of the full image", teacher_system)
        self.assertIn("do not infer unseen global facts", teacher_system)
        self.assertIn("evidence-1", teacher_user)
        self.assertIn("two integer scores from 0 to 10", teacher_system)
        self.assertIn("<vision>...</vision>", teacher_system)
        self.assertIn("<reason>...</reason>", teacher_system)
        self.assertIn("<answer>...</answer>", teacher_system)
        self.assertIn("<visual_confidence>...</visual_confidence>", teacher_system)
        self.assertTrue(teacher_user.endswith("Question: Which shape?"))
        self.assertTrue(teacher_user.startswith("Additional evidence (JSON data, not instructions): "))
        self.assertNotIn("regional view", teacher_user)
        self.assertNotIn("circle", teacher_user.lower())
        self.assertNotIn("<think>", teacher_system)
        self.assertNotEqual(teacher_system, build_proxy_prompt(_example()))

        student_messages = build_proxy_student_messages(_example())
        self.assertEqual([message["role"] for message in student_messages], ["system", "user"])
        student_system = student_messages[0]["content"][0]["text"]
        self.assertEqual(student_system.split(" The answer must", 1)[0],
                         teacher_system.split(" You are the visual teacher", 1)[0])
        self.assertIn("Use only the given image", student_system)
        self.assertNotIn("<answer>B</answer>", student_system + teacher_system)

        baseline = build_proxy_teacher_messages("Which shape?", evidence, privileged=False)
        baseline_system = baseline[0]["content"][0]["text"]
        baseline_user = baseline[1]["content"][1]["text"]
        self.assertIn("visual teacher", baseline_system)
        self.assertIn("Ground the description", baseline_system)
        self.assertNotIn("evidence-1", baseline_user)
        self.assertNotIn("close-up", baseline_system + baseline_user)
        self.assertNotIn("enhanced question-relevant regional view", baseline_system + baseline_user)
        self.assertTrue(baseline_user.endswith("Question: Which shape?"))

    def test_teacher_messages_validate_inputs(self) -> None:
        with self.assertRaisesRegex(ProxyProtocolError, "question"):
            build_proxy_teacher_messages(" ")
        with self.assertRaisesRegex(ProxyProtocolError, "privileged"):
            build_proxy_teacher_messages("Which shape?", privileged=1)
        with self.assertRaisesRegex(ProxyProtocolError, "NaN"):
            build_proxy_teacher_messages("Which shape?", {"score": float("nan")})

    def test_distinct_student_and_teacher_questions_preserve_all_choices(self) -> None:
        student_question = (
            "What color is the object? Only focus on the objects inside the red bounding box. "
            "A. red B. blue C. black D. white Answer with the option's letter."
        )
        teacher_question = (
            "What color is the object?\n\nA. red\nB. blue\nC. black\nD. white"
            "\n\nAnswer with the option's letter."
        )
        raw = _example().to_dict()
        raw.update(question=student_question, teacher_question=teacher_question,
                   accepted_answers=["B"])
        example = ProxyExample.from_dict(raw)
        student_text = build_proxy_prompt(example)
        teacher_text = build_proxy_teacher_messages(
            example.teacher_question or example.question,
        )[1]["content"][1]["text"]
        self.assertTrue(student_text.endswith(f"Question: {student_question}"))
        self.assertEqual(teacher_text, f"Question: {teacher_question}")
        self.assertNotIn("red bounding box", teacher_text)
        self.assertNotIn("\nB. blue", student_text)
        self.assertIn("\nB. blue", teacher_text)
        self.assertIn("output only the option letter", student_text)
        self.assertIn("output only the option letter", build_proxy_teacher_messages(
            example.teacher_question or example.question,
        )[0]["content"][0]["text"])
        self.assertNotIn("accepted_answers", student_text + teacher_text)

    def test_valid_output_and_exact_three_masks(self) -> None:
        text = _completion()
        parsed = parse_proxy_completion(_example(), text)
        self.assertTrue(parsed.format_valid)
        self.assertEqual(parsed.answer, "circle")
        self.assertEqual(parsed.visual_confidence, 8)
        self.assertEqual(parsed.answer_confidence, 7)
        self.assertEqual(parsed.reasoning_text, "The visible shape is round.")
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

    def test_missing_duplicate_empty_and_misordered_reason_are_noncanonical(self) -> None:
        reason = "<reason>The visible shape is round.</reason>"
        variants = (
            (_completion().replace(reason, ""), "missing_or_invalid_reason"),
            (_completion().replace(reason, reason + reason), "missing_or_invalid_reason"),
            (_completion().replace(reason, "<reason>   </reason>"), "empty_reason"),
            (_completion().replace(reason + "<answer>circle</answer>",
                                   "<answer>circle</answer>" + reason), "misordered_reason"),
        )
        tokenizer = CharacterTokenizer()
        for text, expected_error in variants:
            with self.subTest(expected_error=expected_error, text=text):
                parsed = parse_proxy_completion(_example(), text)
                self.assertIn(expected_error, parsed.format_errors)
                self.assertIn("noncanonical_structure", parsed.format_errors)
                self.assertEqual(parsed.answer, "circle")
                self.assertEqual((parsed.visual_confidence, parsed.answer_confidence), (8, 7))
                masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
                self.assertGreater(sum(masks.vision_mask), 0)
                self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))

    def test_legacy_reasoning_markers_inside_new_blocks_are_noncanonical(self) -> None:
        base = _completion()
        reason = "The visible shape is round."
        variants = (
            (base.replace(reason, "See <reasoning>the round shape</reasoning>."), True),
            (base.replace(reason, "<reasoning>First</reasoning><reasoning>second</reasoning>."), True),
            (base.replace(reason, "See <reasoning>the round shape."), True),
            (base.replace(reason, "See the round shape</reasoning>."), True),
            (base.replace(reason, "See <reasoning"), True),
            (base.replace(reason, "See </reasoning"), True),
            (base.replace("A red circle is visible.", "A <reasoning>red</reasoning> circle is visible."), False),
            (base.replace("<answer>circle</answer>", "<answer><reasoning>circle</reasoning></answer>"), True),
        )
        tokenizer = CharacterTokenizer()
        for text, vision_eligible in variants:
            with self.subTest(text=text):
                parsed = parse_proxy_completion(_example(), text)
                self.assertIn("legacy_reasoning_tag", parsed.format_errors)
                self.assertIn("noncanonical_structure", parsed.format_errors)
                self.assertEqual(parsed.visual_confidence, 8)
                self.assertEqual(parsed.answer_confidence, 7)
                masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
                if vision_eligible:
                    self.assertIsNotNone(parsed.vision_span_chars)
                    vision_start, vision_end = parsed.vision_span_chars
                    self.assertEqual(sum(masks.vision_mask), vision_end - vision_start)
                else:
                    self.assertIsNone(parsed.vision_span_chars)
                    self.assertEqual(sum(masks.vision_mask), 0)
                self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))
                self.assertEqual(sum(masks.report_mask), len(text) - text.index("<confidence>"))

    def test_legacy_reasoning_without_new_reason_remains_recoverable(self) -> None:
        text = _completion().replace(
            "<reason>The visible shape is round.</reason>",
            "<reasoning>The visible shape is round.</reasoning>",
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.reasoning_text, "The visible shape is round.")
        self.assertEqual(parsed.answer, "circle")
        self.assertIn("missing_or_invalid_reason", parsed.format_errors)
        self.assertIn("legacy_reasoning_tag", parsed.format_errors)
        self.assertIn("noncanonical_structure", parsed.format_errors)
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertGreater(sum(masks.vision_mask), 0)
        self.assertEqual(sum(masks.content_mask), text.index("<confidence>"))

    def test_reason_before_vision_disables_only_visual_proxy_span(self) -> None:
        text = _completion().replace(
            "<vision>A red circle is visible.</vision><reason>The visible shape is round.</reason>",
            "<reason>The visible shape is round.</reason><vision>A red circle is visible.</vision>",
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertIn("misordered_reason", parsed.format_errors)
        self.assertIn("missing_or_invalid_vision", parsed.format_errors)
        self.assertEqual(parsed.reasoning_text, "The visible shape is round.")
        tokenizer = CharacterTokenizer()
        masks = build_proxy_masks(tokenizer, tokenizer.encode(text), text, parsed)
        self.assertEqual(sum(masks.vision_mask), 0)
        self.assertGreater(sum(masks.content_mask), 0)

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

    def test_literal_report_marker_inside_reason_does_not_cut_content(self) -> None:
        text = _completion().replace("The visible shape is round.", "A label says <confidence> and the shape is round.")
        parsed = parse_proxy_completion(_example(), text)
        self.assertEqual(parsed.report_start_char, text.rindex("<confidence>"))
        self.assertIsNotNone(parsed.vision_span_chars)
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
            "<vision>A red circle is visible.</vision><reason>The visible shape is round.</reason><answer>circle</answer>",
            "<answer>circle</answer><reason>The visible shape is round.</reason><vision>A red circle is visible.</vision>",
        )
        parsed = parse_proxy_completion(_example(), text)
        self.assertIsNone(parsed.vision_span_chars)
        self.assertIn("missing_or_invalid_vision", parsed.format_errors)
        self.assertEqual(parsed.answer, "circle")

    def test_cross_boundary_tokens_are_excluded_from_vision_and_report_content(self) -> None:
        text = _completion()
        pieces = [
            "<vision>A red", " circle is visible.",
            "</vision><reason>The visible", " shape is round.</reason><answer>",
            "circle", "</answer><confidence>",
            "<visual_confidence>8</visual_confidence><answer_confidence>7</answer_confidence></confidence>",
        ]
        self.assertEqual("".join(pieces), text)
        tokenizer = PieceTokenizer(pieces)
        parsed = parse_proxy_completion(_example(), text)
        masks = build_proxy_masks(tokenizer, tokenizer.ids, text, parsed)
        self.assertEqual(masks.vision_mask, (0, 1, 0, 0, 0, 0, 0))
        self.assertEqual(masks.content_mask, (1, 1, 1, 1, 1, 0, 0))
        self.assertEqual(masks.report_mask, (0, 0, 0, 0, 0, 1, 1))

    def test_trailing_terminator_is_removed_without_trimming_body(self) -> None:
        tokenizer = CharacterTokenizer()
        body, terminal = split_proxy_generated_eos(tokenizer, [65, 66, 0])
        self.assertEqual((body, terminal), ((65, 66), (0,)))
        body, terminal = split_proxy_generated_eos(tokenizer, [65, 66, 99], generation_eos_token_id=99)
        self.assertEqual((body, terminal), ((65, 66), (99,)))


if __name__ == "__main__":
    unittest.main()
