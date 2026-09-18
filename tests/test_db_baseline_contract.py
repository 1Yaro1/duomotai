"""CPU-only contract tests; no model download, GAIA CSV, or training."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ts_benchmark.baselines.MindTS.cmc_log_data import (
    BaselineLogView, BaselineWindowDataset)
from ts_benchmark.baselines.MindTS.deterministic_scoring import AbsoluteWindowDataset
from ts_benchmark.baselines.MindTSDBBaselines import MindTSDBBaseline
from scripts.autoresearch.gaia_db_baseline_runner import validate_replay_identity

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    semantic_dim, count_dim = 2, 1

    def __init__(self):
        self.features = np.arange(7056 * 3, dtype=np.float32).reshape(7056, 3)
        self.present = np.ones(7056, dtype=bool)

    def window(self, start, length):
        return self.features[start:start + length].copy(), self.present[start:start + length].astype(np.float32)


class BaselineLogTests(unittest.TestCase):
    def test_null_and_unused_do_not_accept_a_real_store(self):
        for source in ("null", "unused"):
            view = BaselineLogView(source, semantic_dim=2, count_dim=1)
            features, present = view.window(5644, 24)
            np.testing.assert_array_equal(features, np.zeros((24, 3), np.float32))
            np.testing.assert_array_equal(present, np.zeros(24, np.float32))
            with self.assertRaises(ValueError):
                BaselineLogView(source, store=FakeStore(), semantic_dim=2, count_dim=1)

    def test_aligned_view_is_identity(self):
        store = FakeStore()
        view = BaselineLogView("aligned", store=store, semantic_dim=2, count_dim=1)
        values, present = view.window(5644, 24)
        np.testing.assert_array_equal(values, store.features[5644:5668])
        self.assertTrue(present.all())

    def test_shift_is_plus_240_and_circular_inside_each_split(self):
        store = FakeStore()
        view = BaselineLogView("shifted", store=store, semantic_dim=2, count_dim=1)
        for lower, upper in ((0, 4515), (4515, 5644), (5644, 7056)):
            start = upper - 24
            values, _ = view.window(start, 24)
            minute = np.arange(start, upper)
            source = lower + ((minute - lower + 240) % (upper - lower))
            np.testing.assert_array_equal(values, store.features[source])
            self.assertTrue(((source >= lower) & (source < upper)).all())

    def test_shift_cannot_cross_split(self):
        view = BaselineLogView("shifted", store=FakeStore(), semantic_dim=2, count_dim=1)
        with self.assertRaises(ValueError):
            view.window(4500, 24)

    def test_training_dataset_is_step_one_with_absolute_coordinates(self):
        view = BaselineLogView("null", semantic_dim=2, count_dim=1)
        dataset = BaselineWindowDataset(np.zeros((1129, 5)), 4515, view)
        self.assertEqual(len(dataset), 1106)
        self.assertEqual(dataset.first, 4515)
        self.assertEqual(dataset[1][0].shape, (24, 5))
        np.testing.assert_array_equal(dataset[1][3], np.zeros(24, np.float32))

    def test_scoring_dataset_returns_scalar_absolute_start(self):
        view = BaselineLogView("null", semantic_dim=2, count_dim=1)
        trainer = object.__new__(MindTSDBBaseline)
        trainer.baseline_log_view = view
        dataset = trainer.scoring_dataset(np.zeros((1412, 5)), 5644)
        self.assertIsInstance(dataset, AbsoluteWindowDataset)
        self.assertEqual(dataset.absolute_start, 5644)
        self.assertEqual(dataset[0][3], 5644)
        self.assertEqual(dataset[1][3], 5645)
        self.assertTrue(np.isscalar(dataset[1][3]))


class ContractTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "config/autoresearch/baseline_training_contract.json"
        self.contract = json.loads(path.read_text(encoding="utf-8"))

    def test_scope_and_training_are_frozen(self):
        c = self.contract
        self.assertEqual(c["status"], "pretraining_frozen")
        self.assertFalse(c["training_started"])
        self.assertEqual(c["services"], ["GAIA_dbservice1", "GAIA_dbservice2"])
        self.assertEqual(c["splits"]["final_test"], "forbidden")
        self.assertEqual(c["training"]["first_round_seed"], 2021)
        self.assertEqual(c["training"]["maximum_optimizer_updates_per_service"], 1686)
        self.assertTrue(c["training"]["fail_fast"])

    def test_evaluation_contract_hash_and_protocol(self):
        path = ROOT / "config/evaluation_contract_db_overlap_v1.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(actual, self.contract["evaluation"]["contract_sha256"])
        self.assertEqual(self.contract["evaluation"]["score_protocol"], "deterministic_overlap_v1")
        self.assertEqual(self.contract["evaluation"]["threshold_candidate_count"], 61)
        self.assertFalse(self.contract["evaluation"]["point_adjustment"])

    def test_B2_B3_B4_capacity_control_and_sources(self):
        models = self.contract["models"]
        self.assertEqual(models["B2"]["log_source"],
                         "all-zero feature tensor, present=0, learnable missing token")
        self.assertIn("+240-minute", models["B3"]["log_source"])
        self.assertEqual(models["B4"]["log_source"], "correctly aligned log v2")
        self.assertTrue(self.contract["fairness"]["B2_B3_B4_identical_trainable_structure"])
        self.assertFalse(models["B2"]["real_log_read"])
        self.assertTrue(models["B3"]["real_log_read"])
        self.assertTrue(models["B4"]["real_log_read"])

    def test_B5_is_reference_not_retraining_or_causal_proof(self):
        reference = self.contract["models"]["B5"]
        self.assertIn("never retrain", reference["training"])
        self.assertFalse(reference["causal_claim_allowed"])

    def test_checkpoint_replay_allows_only_dataset_interface_and_runner_repair(self):
        stable = {
            "contract_sha256": "contract", "evaluation_contract_sha256": "evaluation",
            "model_id": "B0", "seed": 2021, "prefix_manifest_sha256": "prefix",
            "log_manifest_sha256": "log", "resources": {"config.json": "resource"},
            "source": {
                "ts_benchmark/baselines/MindTS/deterministic_scoring.py": "score-v1",
                "ts_benchmark/baselines/MindTS/models/MindTS_model.py": "model-v1",
                "ts_benchmark/baselines/MindTSDBBaselines.py": "dataset-v1",
                "scripts/autoresearch/gaia_db_baseline_runner.py": "runner-v1",
            },
        }
        repaired = json.loads(json.dumps(stable))
        repaired["source"]["ts_benchmark/baselines/MindTSDBBaselines.py"] = "dataset-v2"
        repaired["source"]["scripts/autoresearch/gaia_db_baseline_runner.py"] = "runner-v2"
        self.assertEqual(validate_replay_identity(stable, repaired), [
            "scripts/autoresearch/gaia_db_baseline_runner.py",
            "ts_benchmark/baselines/MindTSDBBaselines.py",
        ])
        repaired["source"]["ts_benchmark/baselines/MindTS/models/MindTS_model.py"] = "model-v2"
        with self.assertRaisesRegex(ValueError, "Checkpoint-incompatible"):
            validate_replay_identity(stable, repaired)

    def test_B2_B3_B4_have_identical_parameter_topology_and_frozen_qwen(self):
        import importlib
        import torch
        module = importlib.import_module("ts_benchmark.baselines.MindTS.models.MindTS_model")
        from ts_benchmark.baselines.MindTS.MindTS import MINDTSConfig

        class Causal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)

        common = dict(
            batch_size=8, seq_len=24, enc_in_time=2, patch_size=6, stride=6,
            d_model=16, d_ff=8, n_heads=2, e_layers=1, log_input_mode="v2",
            log_semantic_dim=4, log_count_dim=3, contrastive_enabled=False,
            llm_memory_efficient=True)
        modes = ("metric_only_null_log", "metric_log_shifted", "metric_log_simple")
        models = []
        with patch.object(module.AutoConfig, "from_pretrained", return_value=SimpleNamespace()), \
             patch.object(module.AutoTokenizer, "from_pretrained", return_value=lambda *a, **k: None), \
             patch.object(module.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *a, **k: Causal()):
            for mode in modes:
                models.append(module.MINDTSModel(MINDTSConfig(**common, baseline_mode=mode)))
        keys = [tuple(model.state_dict()) for model in models]
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], keys[2])
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])
        for model in models:
            self.assertFalse(any(parameter.requires_grad for parameter in model.model.parameters()))


if __name__ == "__main__":
    unittest.main()
