import unittest

from sure_vl.proxy_protocol import ProxyExample, grade_proxy_answer


class AnswerGradingTests(unittest.TestCase):
    def setUp(self):
        self.example = ProxyExample("one", "train", "student.png", "teacher.png",
                                    "What is it?\nA. boot\nB. harness\nC. sneakers\nD. hat", ["C"])

    def test_correct_choice_text_is_utility_correct_but_format_invalid(self):
        self.assertEqual(grade_proxy_answer(self.example, "C. sneakers"), (True, False))
        self.assertEqual(grade_proxy_answer(self.example, "C"), (True, True))

    def test_ambiguous_or_invented_choice_does_not_earn_correct_reward(self):
        for value in ("C or B", "C. harness", "C. sneakers or B", "B. harness"):
            self.assertFalse(grade_proxy_answer(self.example, value)[0])

    def test_missing_answer_has_known_incorrect_event(self):
        self.assertEqual(grade_proxy_answer(self.example, None), (False, False))

    def test_open_answers_keep_exact_normalized_semantics(self):
        example = ProxyExample("number", "train", "s.png", "t.png", "How many?", ["2"])
        self.assertEqual(grade_proxy_answer(example, "2"), (True, True))
        self.assertEqual(grade_proxy_answer(example, "2 or 3"), (False, True))
