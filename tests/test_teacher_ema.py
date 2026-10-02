import unittest
from types import SimpleNamespace

from sure_vl.teacher_ema import EMATeacher, OptimizerEvidenceCallback

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EMATests(unittest.TestCase):
    def test_nonfinite_gradients_are_rejected_before_optimizer_update(self):
        student = torch.nn.Linear(1, 1, bias=False)
        student.weight.grad = torch.full_like(student.weight, float("nan"))
        initial = student.weight.detach().clone()
        callback = OptimizerEvidenceCallback(
            SimpleNamespace(model=student), teacher_mode="fixed", ema_decay=0.99
        )
        with self.assertRaisesRegex(RuntimeError, "nonfinite student gradients"):
            callback.on_pre_optimizer_step(None, None, None)
        self.assertTrue(torch.equal(initial, student.weight))
        self.assertEqual(callback.successful_updates, 0)

    def test_finite_gradients_pass_the_update_gate(self):
        student = torch.nn.Linear(1, 1, bias=False)
        student.weight.grad = torch.ones_like(student.weight)
        callback = OptimizerEvidenceCallback(
            SimpleNamespace(model=student), teacher_mode="fixed", ema_decay=0.99
        )
        control = object()
        self.assertIs(callback.on_pre_optimizer_step(None, None, control), control)

    def test_ema_uses_successive_fp32_masters_and_frozen_teacher(self):
        student = torch.nn.Linear(1, 1, bias=False).half()
        teacher = torch.nn.Linear(1, 1, bias=False).half()
        with torch.no_grad():
            student.weight.fill_(2)
            teacher.weight.fill_(0)
        ema = EMATeacher(student, teacher, decay=0.5)
        ema.update()
        self.assertEqual(teacher.weight.item(), 1)
        ema.update()
        self.assertEqual(teacher.weight.item(), 1.5)
        self.assertEqual(ema.masters["weight"].dtype, torch.float32)
        self.assertFalse(teacher.weight.requires_grad)
        self.assertFalse(teacher.training)
        self.assertEqual(ema.update_count, 2)


class OptimizerEvidenceTests(unittest.TestCase):
    def test_skipped_step_does_not_count_as_update_or_update_teacher(self):
        accelerator = SimpleNamespace(optimizer_step_was_skipped=False)
        callback = OptimizerEvidenceCallback(
            SimpleNamespace(accelerator=accelerator), teacher_mode="fixed", ema_decay=0.99
        )
        state = SimpleNamespace(global_step=1)
        callback.on_step_end(None, state, None)
        accelerator.optimizer_step_was_skipped = True
        state.global_step = 2
        callback.on_step_end(None, state, None)
        self.assertEqual(callback.summary()["optimizer_attempted_steps"], 2)
        self.assertEqual(callback.summary()["optimizer_successful_updates"], 1)
        self.assertEqual(callback.summary()["optimizer_skipped_updates"], 1)


if __name__ == "__main__":
    unittest.main()
