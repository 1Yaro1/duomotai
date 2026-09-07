"""Synthetic data/interface regressions; no GAIA training, labels, or test files."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_file("prep_v2", "scripts/autoresearch/prepare_gaia_log_features_v2.py")
logs = load_file("logs_v2", "ts_benchmark/baselines/MindTS/cmc_log_data.py")


class SyntheticEmbedding:
    """Fixture vectors, explicitly NOT BERT or a model-quality experiment."""
    def __call__(self, texts):
        return np.array([[len(t), t.count("error"), 1., 0.] for t in texts], dtype=np.float32)

    def provenance(self):
        return {"type": "synthetic_fixture_only", "hidden_size": 4}


class LogV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2021)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["id", "datetime", "service", "message"])
        i = 0
        for service in prep.SERVICES:
            for minute, message in [(0, 'INFO accepted request 10, "ok"\nsecond line'),
                                    (1, "INFO accepted request 20"),
                                    (4514, "INFO accepted request 30"),
                                    (4515, "ERROR completely new exception trace with unique tokens"),
                                    (5644, "ERROR completely new exception trace with unique tokens"),
                                    (7055, "INFO accepted request 40"),
                                    (7056, "synthetic forbidden-period canary"),
                                    (2, "trigger a high memory program")]:
                stamp = pd.Timestamp("2021-08-24") + pd.Timedelta(minutes=minute)
                raw = f"{stamp:%Y-%m-%d %H:%M:%S},125 | INFO | host | {service} | code.py | trace | {message}"
                writer.writerow([i, f"{stamp:%Y-%m-%d}", service, raw])
                i += 1
        writer.writerow([i, "2021-08-24", "dbservice1", ""])
        cls.archive = cls.root / "fixture.zip"
        with zipfile.ZipFile(cls.archive, "w") as z:
            z.writestr(prep.MEMBER, text.getvalue())
        cls.source_hash = prep.sha256(cls.archive)
        cls.raw, cls.cache = cls.root / "raw", cls.root / "cache"
        prep.extract(cls.archive, cls.raw)
        prep.build(cls.raw, cls.cache, SyntheticEmbedding())
        cls.store = logs.LogV2Store(cls.cache, "GAIA_dbservice1")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_source_unchanged(self):
        self.assertEqual(prep.sha256(self.archive), self.source_hash)

    def test_existing_output_refused(self):
        with self.assertRaises(FileExistsError):
            prep.extract(self.archive, self.raw)

    def test_multiline_and_quotes_preserved(self):
        with prep.raw_connection(self.raw / "events.sqlite") as db:
            value = db.execute("SELECT message FROM events WHERE source_order=0").fetchone()[0]
        self.assertIn('10, "ok"\nsecond line', value)

    def test_empty_source_record_preserved_without_invented_minute(self):
        with prep.raw_connection(self.raw / "events.sqlite") as db:
            row = db.execute("SELECT service,source_date,message,reason FROM empty_messages").fetchone()
        self.assertEqual(row, ("dbservice1", "2021-08-24", "", "empty_message_no_event_timestamp"))
        self.assertEqual(int(self.store.counts.sum()), 6)

    def test_timestamp_whitespace_and_nonempty_missing(self):
        self.assertEqual(prep.event_position("\n 2021-08-24 00:01:30,500 | INFO"), (1, 30.5 / 60))
        with self.assertRaises(ValueError):
            prep.event_position("nonempty exception without timestamp")

    def test_no_test_rows_materialized(self):
        with prep.raw_connection(self.raw / "events.sqlite") as db:
            self.assertEqual(db.execute("SELECT MAX(minute) FROM events").fetchone()[0], 7055)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE message LIKE '%canary%'").fetchone()[0], 0)

    def test_control_raw_preserved_not_used_as_feature(self):
        with prep.raw_connection(self.raw / "events.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE excluded=1").fetchone()[0], 10)
        self.assertEqual(int(self.store.counts[2].sum()), 0)

    def test_validation_does_not_fit_vocabulary(self):
        self.assertEqual(self.store.counts[4515, 0], 1)
        self.assertEqual(self.store.counts[5644, 0], 1)
        vocab = (self.cache / "GAIA_dbservice1.vocabulary.json").read_text()
        self.assertNotIn("unique tokens", vocab)

    def test_train_only_count_scaler(self):
        np.testing.assert_allclose(self.store.mean, np.log1p(self.store.counts[:4515]).astype(np.float32).mean(0))
        self.assertEqual(self.store.mean[0], 0)

    def test_all_services_and_minutes(self):
        for service in logs.SERVICES:
            store = logs.LogV2Store(self.cache, service)
            self.assertEqual(len(store.counts), 7056)
            self.assertEqual(int(store.counts.sum()), 6)

    def test_no_character_truncation(self):
        value = "long body " * 500 + "TAIL_SENTINEL"
        self.assertTrue(prep.normalize(value).endswith("TAIL_SENTINEL"))

    def test_missing_minute_preserved(self):
        features, present = self.store.window(10, 24)
        self.assertEqual(features.shape[0], 24)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(present.sum(), 0)
        self.assertEqual(features[:, :self.store.semantic_dim].sum(), 0)

    def data(self, first, n):
        return pd.DataFrame(np.zeros((n, 2), dtype=np.float32),
                            index=pd.date_range(logs.START + pd.Timedelta(minutes=first), periods=n, freq="min"))

    def test_legacy_window_counts_and_tail_preserved(self):
        val = self.data(5644, 1412)
        sliding = logs.LogV2Dataset(val, self.store, mode="test")
        nonoverlap = logs.LogV2Dataset(val, self.store, mode="thre")
        self.assertEqual(len(sliding), 1389)
        self.assertEqual(len(nonoverlap), 58)
        self.assertEqual(len(nonoverlap) * 24, 1392)

    def test_window_shapes(self):
        loader = logs.log_v2_provider(self.data(0, 48), None, batch_size=2, store=self.store)
        metric, feature, present, dummy = next(iter(loader))
        self.assertEqual(metric.shape, (2, 24, 2))
        self.assertEqual(feature.shape, (2, 24, self.store.semantic_dim + self.store.count_dim))
        self.assertEqual(present.shape, (2, 24))
        self.assertEqual(dummy.sum(), 0)

    def test_actual_strategy_integer_index_mapping(self):
        for first, mode in ((0, "train"), (4515, "val"), (5644, "test"), (5644, "thre")):
            dated = self.data(first, 48)
            numbered = dated.copy()
            numbered.index = pd.RangeIndex(first, first + 48)
            a = logs.LogV2Dataset(dated, self.store, mode=mode)
            b = logs.LogV2Dataset(numbered, self.store, mode=mode)
            for left, right in zip(a[0], b[0]):
                np.testing.assert_array_equal(left, right)

    def test_reset_validation_slice_refused(self):
        bad = self.data(5644, 48).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "mode"):
            logs.LogV2Dataset(bad, self.store, mode="test")

    @unittest.skipUnless((ROOT / "ts_benchmark/data/utils.py").exists(), "Requires actual project dataframe converter")
    def test_numeric_prefix_never_parses_test_or_unrequested_labels(self):
        path = self.root / "synthetic_numeric.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "data", "cols"])
            for name in ("metric_a", "metric_b", "label"):
                for row in range(1, 10081):
                    value = "MUST_NOT_PARSE_TEST" if row > 7056 else ("MUST_NOT_PARSE_LABEL" if name == "label" else row / 100)
                    writer.writerow([row, value, name])
        result = logs.load_numeric_prefix(path, 2)
        self.assertEqual(result.shape, (7056, 2))
        self.assertEqual(result.iloc[0, 0], 0.01)
        self.assertEqual(result.iloc[-1, -1], 70.56)

    def test_validation_entry_defaults_to_checks_not_training(self):
        runner = load_file("runner_v2", "scripts/autoresearch/gaia_log_v2_validation_runner.py")
        argv = ["runner", "--cache-dir", str(self.cache)]
        with patch.object(runner.sys, "argv", argv), patch.object(runner, "configuration", return_value=({"services": []}, {})), \
             patch.object(runner, "run_guards") as guards, patch.object(runner, "check_inputs") as checks, \
             patch.object(runner, "worker", side_effect=AssertionError("Training must not run")) as worker, \
             patch.object(runner.os, "chdir"):
            runner.main()
        guards.assert_called_once()
        checks.assert_called_once()
        worker.assert_not_called()

    def test_bad_index_and_split_crossings_refused(self):
        for first in (4500, 5630, 7050, 7056):
            with self.assertRaises(ValueError):
                logs.LogV2Dataset(self.data(first, 30), self.store)
        for data in (self.data(0, 48).iloc[::-1], self.data(0, 48).iloc[::2]):
            with self.assertRaises(ValueError):
                logs.LogV2Dataset(data, self.store)

    def test_out_of_scope_store_window_refused(self):
        for first in (-1, 7056, 7040):
            with self.assertRaises(ValueError):
                self.store.window(first, 24)

    def test_encoder_forward_backward_and_missing(self):
        encoder = logs.MinuteLogEncoder(self.store.semantic_dim, self.store.count_dim, 16)
        for first in (0, 100):
            features, present = self.store.window(first, 24)
            output = encoder(torch.tensor(features)[None], torch.tensor(present)[None])
            self.assertEqual(output.shape, (1, 24, 16))
            self.assertTrue(torch.isfinite(output).all())
            output[..., 0].sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None))

    def test_encoder_save_restore(self):
        one = logs.MinuteLogEncoder(4, self.store.count_dim, 16)
        two = logs.MinuteLogEncoder(4, self.store.count_dim, 16)
        two.load_state_dict(one.state_dict())
        feature, present = self.store.window(0, 24)
        args = torch.tensor(feature)[None], torch.tensor(present)[None]
        torch.testing.assert_close(one(*args), two(*args), rtol=0, atol=0)

    def test_numeric_channel_batch_alignment(self):
        value = torch.arange(2).reshape(2, 1, 1).repeat(1, 24, 8)
        self.assertEqual(value.repeat_interleave(3, dim=0)[:, 0, 0].tolist(), [0, 0, 0, 1, 1, 1])

    def test_legacy_provider_dispatch_unchanged(self):
        tree = ast.parse((ROOT / "ts_benchmark/baselines/MindTS/MindTS.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MindTS")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_multi_data_provider")
        env = {"anomaly_detection_multi_data_provider": lambda *a, **kw: (a, kw)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dispatch", "exec"), env)
        obj = type("Legacy", (), {"log_v2_store": None})()
        self.assertEqual(env["_multi_data_provider"](obj, "x", mode="val"), (("x",), {"mode": "val"}))

    def test_loss_source_unchanged(self):
        source = (ROOT / "ts_benchmark/baselines/MindTS/MindTS.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Hash source segments: ast.dump itself differs between Python 3.10/3.13.
        hashes = {"clip_loss": "c62fe3daa806fc1449aa4ac569009f5b1ce618ddbb511de0f1f96cfff7019ca1",
                  "Bottleneck_loss": "3ebd97bbb29367817edab60667b9c3f655414aab8fcb719dfe13ebd684545845",
                  "CMCInspiredLoss": "6728076a01df2d6cbac429385ed7e18a6fb9b81ee2f696a56347c822bcd35023"}
        actual = {n.name: hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()
                  for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in hashes}
        self.assertEqual(actual, hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
