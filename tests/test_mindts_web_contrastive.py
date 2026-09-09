"""Fast CPU tests; no models downloaded, no GAIA/test data loaded."""
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = source_module("contrast_test", "ts_benchmark/baselines/MindTS/contrastive_loss.py")
K = source_module("checkpoint_test", "ts_benchmark/baselines/MindTS/checkpointing.py")


class ContrastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)

    def test_overlap_and_boundaries(self):
        starts = torch.tensor([0, 1, 12, 24, 48])
        mask = C.negative_mask(starts)
        self.assertEqual(mask[0].tolist(), [False, False, False, True, True])
        self.assertTrue(torch.equal(mask, mask.T))
        self.assertFalse(mask.diag().any())
        for value in (-1, 4492, 4515, 5644):
            with self.assertRaises(ValueError):
                C.negative_mask(torch.tensor([value]))

    def test_matches_cross_entropy_without_exclusions(self):
        a, b = torch.randn(8, 12), torch.randn(8, 12)
        mask = ~torch.eye(8, dtype=torch.bool)
        loss, count = C.directional_nce(a, b, mask)
        expected = torch.nn.functional.cross_entropy(
            torch.nn.functional.normalize(a, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).T / .2,
            torch.arange(8))
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 8)

    def test_masked_candidates_have_no_gradient(self):
        a, b = torch.randn(3, 5, requires_grad=True), torch.randn(3, 5, requires_grad=True)
        mask = torch.zeros(3, 3, dtype=torch.bool)
        mask[0, 1] = True
        loss, count = C.directional_nce(a, b, mask)
        loss.backward()
        self.assertEqual(count, 1)
        self.assertEqual(float(b.grad[2].abs().sum()), 0.)

    def test_singleton_differentiable_zero(self):
        a = torch.randn(1, 5, requires_grad=True)
        loss, count = C.directional_nce(a, a, torch.zeros(1, 1, dtype=torch.bool))
        loss.backward()
        self.assertEqual(count, 0)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_missing_logs_keep_metric_loss(self):
        views = {k: torch.randn(8, 5, requires_grad=True) for k in ("ma", "mb", "la", "lb")}
        terms = C.dual_contrast(views, torch.arange(8) * 24, torch.zeros(8, 24))
        self.assertEqual(terms["valid_metric"], 8)
        self.assertEqual(terms["valid_log"], 0)
        self.assertEqual(terms["valid_inter"], 0)
        self.assertGreater(float(terms["intra_metric"]), 0.)

    def test_heads_and_both_encoders_receive_gradients(self):
        metric_encoder = torch.nn.Linear(3, 12)
        log_encoder = torch.nn.Linear(3, 12)
        mh, lh = C.WindowContrastHead(4, 12), C.WindowContrastHead(24, 12)
        ma, mb = mh(metric_encoder(torch.randn(8, 4, 3)))
        la, lb = lh(log_encoder(torch.randn(8, 24, 3)))
        terms = C.dual_contrast(dict(ma=ma, mb=mb, la=la, lb=lb), torch.arange(8) * 24, torch.ones(8, 24))
        (terms["intra_metric"] + terms["intra_log"] + terms["inter"]).backward()
        for module in (metric_encoder, log_encoder, mh, lh):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
        self.assertFalse(torch.equal(ma, mb))
        mh.eval()
        a, b = mh(torch.randn(1, 4, 12))
        self.assertTrue(torch.equal(a, b))

    def test_joint_permutation_invariance(self):
        views = {k: torch.randn(8, 5) for k in ("ma", "mb", "la", "lb")}
        starts, present = torch.arange(8) * 20, torch.ones(8, 24)
        one = C.dual_contrast(views, starts, present)
        p = torch.randperm(8)
        two = C.dual_contrast({k: v[p] for k, v in views.items()}, starts[p], present[p])
        for key in ("intra_metric", "intra_log", "inter"):
            torch.testing.assert_close(one[key], two[key])

    def test_invalid_settings(self):
        x = torch.randn(2, 3)
        with self.assertRaises(ValueError):
            C.directional_nce(x, x, torch.eye(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            C.directional_nce(x, x, ~torch.eye(2, dtype=torch.bool), temperature=0)


class CheckpointTests(unittest.TestCase):
    def test_rng_roundtrip(self):
        random.seed(2021); np.random.seed(2021); torch.manual_seed(2021)
        state = K.capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(7))
        K.restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(7))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_full_optimizer_and_rng_resume_exact(self):
        torch.manual_seed(2021)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        data = torch.randn(4, 4)
        def step(m, opt):
            opt.zero_grad(); loss = m(data).square().mean(); loss.backward(); opt.step()
            return loss.detach()
        step(model, optimizer)
        payload = dict(schema_version=1, identity={"source": "fixture"}, model=K.cpu_tree(model.state_dict()),
                       optimizer=K.cpu_tree(optimizer.state_dict()), rng=K.capture_rng(),
                       order=K.epoch_order(31, 0), next_offset=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            K.atomic_save(payload, path)
            expected_loss = step(model, optimizer)
            expected_weights = K.cpu_tree(model.state_dict())
            restored = K.load_trusted(path, {"source": "fixture"})
            resumed = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 2))
            resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.123)
            resumed.load_state_dict(restored["model"])
            resumed_optimizer.load_state_dict(restored["optimizer"])
            K.restore_rng(restored["rng"])
            self.assertTrue(torch.equal(expected_loss, step(resumed, resumed_optimizer)))
            for key, value in resumed.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[key]))
            self.assertEqual(sorted(restored["order"]), list(range(31)))
            self.assertEqual(restored["next_offset"], 8)
            with self.assertRaises(ValueError):
                K.load_trusted(path, {"source": "changed"})
            self.assertFalse(list(Path(folder).glob("*.tmp")))

    def test_cpu_state_is_not_aliased(self):
        original = {"x": torch.ones(3, requires_grad=True)}
        copy = K.cpu_tree(original)
        original["x"].data.zero_()
        self.assertEqual(float(copy["x"].sum()), 3.)
        self.assertFalse(copy["x"].requires_grad)

    def test_sampler_full_coverage_and_stable(self):
        self.assertEqual(K.epoch_order(4492, 0), K.epoch_order(4492, 0))
        self.assertNotEqual(K.epoch_order(4492, 0), K.epoch_order(4492, 1))
        self.assertEqual(sorted(K.epoch_order(4492, 0)), list(range(4492)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
