import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ts_benchmark.baselines.MindTS.deterministic_scoring import (
    MASK_BANK, AbsoluteWindowDataset, apply_fixed_patch_mask,
    score_series, score_threshold_and_prediction)


class Logs:
    def window(self, start, size):
        return np.zeros((size, 3), np.float32), np.ones(size, np.float32)


class Toy(torch.nn.Module):
    seq_len, patch_num, patch_size, stride = 24, 4, 6, 6
    def forward(self, x, log, present, fixed_patch_mask=None, deterministic_gate=False):
        assert deterministic_gate
        mask = torch.tensor(np.repeat(fixed_patch_mask, 6), device=x.device)
        # Visible positions have arbitrarily large error; masked error is one.
        return (x + torch.where(mask[None, :, None], 1., 10000.),)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.model = Toy().eval()
        self.dataset = AbsoluteWindowDataset(np.zeros((1412, 2)), 5644, Logs())

    def score(self):
        return score_series(self.model, self.dataset, device="cpu")

    def test_score_is_bitwise_deterministic(self):
        first = self.score()
        torch.rand(123)
        np.random.rand(123)
        with patch("torch.rand", side_effect=AssertionError("eval RNG")), patch(
                "torch.nn.functional.gumbel_softmax", side_effect=AssertionError("eval Gumbel")):
            second = self.score()
        for key in first:
            self.assertEqual(first[key].tobytes(), second[key].tobytes())

    def test_validation_score_length_is_1412(self):
        self.assertEqual(self.score()["score"].shape, (1412,))
        self.assertEqual(len(self.dataset), 1389)
        self.assertEqual(self.dataset[1388][3], 7032)

    def test_all_points_have_positive_coverage(self):
        result = self.score()
        self.assertTrue((result["coverage"] > 0).all())
        np.testing.assert_array_equal(result["mask_coverage"], result["coverage"] * 3)

    def test_no_tail_padding(self):
        with patch("numpy.pad", side_effect=AssertionError("padding forbidden")):
            result = self.score()
        self.assertEqual(result["coverage"][-1], 1)
        self.assertEqual(result["score"][-1], 1)

    def test_threshold_and_prediction_use_same_score_function(self):
        training = AbsoluteWindowDataset(np.zeros((4515, 2)), 0, Logs())
        with patch("ts_benchmark.baselines.MindTS.deterministic_scoring.score_series",
                   wraps=score_series) as shared:
            train, val, ratios, thresholds, predicted = score_threshold_and_prediction(
                self.model, training, self.dataset, device="cpu")
        self.assertEqual(shared.call_count, 2)
        self.assertEqual(len(ratios), 61)
        np.testing.assert_array_equal(predicted, val["score"][None, :] > thresholds[:, None])

    def test_masked_score_ignores_visible_positions(self):
        np.testing.assert_array_equal(self.score()["channel_error"], np.ones((1412, 2)))

    def test_six_mask_bank_and_patch_mask(self):
        self.assertEqual(len(set(MASK_BANK)), 6)
        self.assertTrue(np.all(np.sum(MASK_BANK, axis=0) == 3))
        for mask in MASK_BANK:
            out = apply_fixed_patch_mask(torch.ones(3, 4, 5), mask)
            self.assertEqual((out == 0).sum().item(), 30)

    def test_split_crossing_rejected(self):
        with self.assertRaises(ValueError):
            AbsoluteWindowDataset(np.zeros((30, 2)), 5640, Logs())

    def test_prefix_skips_unparseable_suffix_without_decoding(self):
        path = Path(__file__).resolve().parents[1] / "scripts/autoresearch/extract_db_prefix_once.py"
        spec = importlib.util.spec_from_file_location("prefix_once", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = b"date,data,cols\n1,2.5,metric\n7057,\xff,not-a-label-to-decode\n2,1,label\n"
        self.assertEqual(module.select_prefix_stream(io.BytesIO(raw)),
                         b"date,data,cols\n1,2.5,metric\n2,1,label\n")


if __name__ == "__main__":
    unittest.main()
