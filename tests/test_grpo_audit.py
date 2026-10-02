"""Small checks for exact-ID generation/forward comparison mathematics."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from scripts.audit_grpo_sampling import compare_generation_and_forward_logits


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GRPOSamplingAuditTests(unittest.TestCase):
    def test_identical_causal_distributions_have_zero_gap(self):
        step0 = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
        step1 = torch.tensor([[0.1, 0.8, 0.2], [5.0, -5.0, 0.0]])
        forward = torch.stack((step0, step1), dim=1)
        ids = torch.tensor([[0, 1], [1, 0]])
        mask = torch.tensor([[1, 1], [1, 0]])
        result = compare_generation_and_forward_logits((step0, step1), forward, ids, mask)
        self.assertEqual(result["compared_tokens"], 3)
        self.assertAlmostEqual(result["max_abs_sample_logprob_delta"], 0.0)
        self.assertAlmostEqual(result["max_jensen_shannon_nats"], 0.0, places=6)

    def test_mismatch_on_sampled_distribution_is_detected(self):
        generated = torch.tensor([[4.0, 0.0]])
        forward = torch.tensor([[[0.0, 4.0]]])
        result = compare_generation_and_forward_logits(
            (generated,), forward, torch.tensor([[0]]), torch.tensor([[1]])
        )
        self.assertGreater(result["max_abs_sample_logprob_delta"], 3.0)
        self.assertGreater(result["max_jensen_shannon_nats"], 0.4)

    def test_cannot_silently_compare_no_real_ids(self):
        with self.assertRaisesRegex(ValueError, "no sampled"):
            compare_generation_and_forward_logits(
                (torch.zeros((1, 3)),), torch.zeros((1, 1, 3)),
                torch.tensor([[0]]), torch.tensor([[0]]),
            )


if __name__ == "__main__":
    unittest.main()
