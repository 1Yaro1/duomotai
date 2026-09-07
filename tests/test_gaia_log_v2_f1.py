"""Synthetic metric/selection regressions: never reads GAIA observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.autoresearch import gaia_log_v2_f1_runner as runner
from ts_benchmark.evaluation.metrics import classification_metrics_label as metrics


class F1Tests(unittest.TestCase):
    def test_f1_not_affiliation_and_first_tie(self):
        rows = [{"anomaly_ratio": 0.5, "F1": .5, "affiliation_f": .99},
                {"anomaly_ratio": 1, "F1": .8, "affiliation_f": .6},
                {"anomaly_ratio": 1.5, "F1": .8, "affiliation_f": .9}]
        self.assertIs(runner.best_row(rows, [.5, 1, 1.5]), rows[1])
        with self.assertRaises(ValueError):
            runner.best_row(rows[::-1], [.5, 1, 1.5])
        with self.assertRaises(ValueError):
            runner.best_row(rows[:2], [.5, 1, 1.5])
        rows[0]["F1"] = float("nan")
        with self.assertRaises(ValueError):
            runner.best_row(rows, [.5, 1, 1.5])

    def test_metric_equivalence_and_score_cache(self):
        actual = np.zeros(1412)
        actual[10:13] = 1
        actual[100:104] = 1
        score = np.linspace(.001, .6, 1412)
        score[actual == 1] = [.8, .7, .5, .9, .4, .95, .3]
        first, second = (score > .55).astype(float), (score > .75).astype(float)
        original = (actual.copy(), score.copy(), first.copy(), second.copy())
        ev = runner.RecordingEvaluator()
        with patch.object(metrics, "VUS_PR", wraps=metrics.VUS_PR) as pr:
            values, log = ev.evaluate_with_log(actual, first, score)
            next_values, _ = ev.evaluate_with_log(actual, second, score)
            self.assertEqual(pr.call_count, 1)
        self.assertEqual(log, "")
        for index, (name, function) in enumerate(runner.METRICS.items()):
            self.assertAlmostEqual(values[index], getattr(metrics, function)(actual, first, score), places=12)
            self.assertAlmostEqual(next_values[index], getattr(metrics, function)(actual, second, score), places=12)
        for before, after in zip(original, (actual, score, first, second)):
            np.testing.assert_array_equal(before, after)
        with self.assertRaises(ValueError):
            ev.evaluate_with_log(actual, second, score + .001)
        with self.assertRaises(ValueError):
            ev.evaluate_with_log(actual[:-1], second[:-1], score[:-1])

    def test_ten_service_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            services = [{"name": f"synthetic-{i}"} for i in range(10)]
            for service in services:
                rows = []
                for ratio, f1, aff in [(.5, .5, .99), (1, .8, .7)]:
                    row = dict(service=service["name"], anomaly_ratio=ratio, energy_threshold=.123,
                               elapsed_seconds=1, **{k: .6 for k in runner.METRICS})
                    row.update(F1=f1, affiliation_f=aff)
                    rows.append(row)
                runner.write_csv(directory / (service["name"] + ".csv"), rows)
            runner.summarize(directory, {"services": services, "anomaly_ratios": [.5, 1]})
            result = json.loads((directory / "summary.json").read_text())
            self.assertAlmostEqual(result["mean_best_f1"], .8)
            self.assertAlmostEqual(result["macro_average"]["affiliation_f"], .7)
            self.assertEqual((directory / "summary.log").read_text().splitlines()[-1], '{"mean_best_f1": 0.8000}')

    def test_active_configuration(self):
        self.assertEqual(runner.selection_config()["selection_metric"], "f_score")


if __name__ == "__main__":
    unittest.main()
