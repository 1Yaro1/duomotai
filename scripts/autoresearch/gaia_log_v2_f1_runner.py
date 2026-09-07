"""Versioned log-v2 validation: select ordinary F1, record eight existing metrics.

No changes to training, loss, split, prediction, padding, or threshold search.
Default is preflight only. Completed-service arrays are retained for auditing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.autoresearch import gaia_log_v2_validation_runner as base

METRICS = {"affiliation_f": "affiliation_f", "PR": "precision", "RC": "recall",
           "F1": "f_score", "AUC": "auc_roc", "AP": "auc_pr",
           "V-ROC": "VUS_ROC", "V-PR": "VUS_PR"}
SCORE_METRICS = ("AUC", "AP", "V-ROC", "V-PR")
FIELDS = ["service", "anomaly_ratio", "energy_threshold", *METRICS, "elapsed_seconds"]


def selection_config():
    value = base.read_json(ROOT / "config/autoresearch/gaia_log_v2_f1.json")
    expected = {"selection_metric": "f_score", "reported_selection_metric": "F1",
                "point_adjustment": False, "tie_break": "first_in_frozen_ratio_order",
                "validation_interval": [5644, 7056], "seed": 2021,
                "test_evaluated": False, "service_weighting": "equal", "metrics": METRICS}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Point-F1 selection configuration differs from approved protocol")
    return value


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows, ratios):
    if [float(r["anomaly_ratio"]) for r in rows] != list(map(float, ratios)):
        raise ValueError("Missing, reordered or duplicate threshold candidates")
    if not rows or any(not math.isfinite(float(r["F1"])) for r in rows):
        raise ValueError("Incomplete ordinary F1 values")
    # Python max is stable: first candidate wins ties, never affiliation_f.
    return max(rows, key=lambda r: float(r["F1"]))


class RecordingEvaluator:
    """Delegate formulas to the project, capture unchanged evaluation inputs.

The four score-based functions ignore predicted labels. Compute once, and
assert identical labels/scores before reusing across the threshold candidates.
"""
    metric_names = list(METRICS)

    def __init__(self):
        self.actual = None
        self.score = None
        self.predictions = []
        self.continuous = {}

    def default_result(self):
        return [float("nan")] * len(METRICS)

    def evaluate_with_log(self, actual, predicted, another, **kwargs):
        import numpy as np
        from ts_benchmark.evaluation.metrics import classification_metrics_label as metrics
        if actual.shape != (1412,) or predicted.shape != actual.shape or another.shape != actual.shape:
            raise ValueError("Expected unchanged 1412-point internal-validation evaluation arrays")
        if not all(np.isfinite(x).all() for x in (actual, predicted, another)):
            raise ValueError("Non-finite evaluation arrays")
        if self.actual is None:
            self.actual, self.score = actual.copy(), another.copy()
            for key in SCORE_METRICS:
                print(json.dumps({"event": "score_metric_start", "metric": key}), flush=True)
                self.continuous[key] = float(getattr(metrics, METRICS[key])(actual, predicted, another))
        elif not np.array_equal(actual, self.actual) or not np.array_equal(another, self.score):
            raise ValueError("Labels or continuous scores changed across thresholds")
        self.predictions.append(predicted.copy())
        values = [self.continuous[k] if k in self.continuous else
                  float(getattr(metrics, name)(actual, predicted, another)) for k, name in METRICS.items()]
        return values, ""


def worker(cache, service, protocol, baseline, output):
    import numpy as np
    import pandas as pd
    from ts_benchmark.baselines.MindTSLogged import MindTSLogged
    from ts_benchmark.data.data_pool import DataPool
    from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
    from ts_benchmark.data.dataset import Dataset
    from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
    from ts_benchmark.models import ModelFactory
    from ts_benchmark.utils.random_utils import fix_random_seed

    name = service["name"]
    threshold_path = output / f"{name}.thresholds.csv"
    os.environ["MINDTS_THRESHOLD_LOG_PATH"] = str(threshold_path)
    logs = base.load_logs()
    series = logs.load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"],
                                     int(service["channels"]), include_label=True)
    if len(series) != 7056:
        raise ValueError("Validation-only numeric prefix length mismatch")
    text = pd.DataFrame({"channel1": [""] * len(series)}, index=series.index)
    source = LocalAnomalyDetectDataSource()  # Metadata only, not test data.
    metadata = source.dataset.metadata.loc[[service["series"]]].copy(deep=True)
    for key, value in {"train_lens": 5644, "time_steps": 7056, "total_len": 7056, "train/total": 5644/7056}.items():
        metadata.loc[service["series"], key] = value
    pool = Dataset()
    pool.set_data(data_dict={service["series"]: series}, covariate_dict={}, metadata=metadata)
    pool.update_multi_data({}, {service["text"]: text}, {})
    DataPool().set_pool(pool)
    params = dict(baseline, log_input_mode="v2", log_v2_cache_dir=str(cache),
                  log_v2_service=name, enc_in_time=int(service["channels"]),
                  anomaly_ratio=list(protocol["anomaly_ratios"]))
    fix_random_seed(2021)
    evaluator = RecordingEvaluator()
    strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label"}, evaluator)
    factory = ModelFactory("MindTSLogged", MindTSLogged, params)
    print(json.dumps({"event": "service_start", "service": name, "selection_metric": "F1"}), flush=True)
    started = time.monotonic()
    raw = strategy.multi_execute(service["series"], service["text"], factory)
    elapsed = time.monotonic() - started
    if len(raw) != 61 or len(evaluator.predictions) != 61:
        raise RuntimeError("Incomplete service evaluation (no retry): " + repr(raw))
    # Preserve inputs before any result validation: failed aggregation is auditable.
    np.savez_compressed(output / f"{name}.validation_arrays.npz", actual=evaluator.actual,
                        score=evaluator.score, predicted=np.stack(evaluator.predictions),
                        anomaly_ratio=np.asarray(protocol["anomaly_ratios"]))
    with threshold_path.open(newline="", encoding="utf-8") as handle:
        thresholds = list(csv.DictReader(handle))
    if [float(t["anomaly_ratio"]) for t in thresholds] != list(map(float, protocol["anomaly_ratios"])):
        raise ValueError("Threshold log order differs from frozen search")
    rows = []
    for record, threshold in zip(raw, thresholds):
        r = dict(zip(strategy.field_names, record))
        if float(r["typical_anomaly_ratio"]) != float(threshold["anomaly_ratio"]) or r.get("log_info"):
            raise ValueError("Raw result / threshold mismatch or evaluator error: " + repr(r))
        row = dict(service=name, anomaly_ratio=float(threshold["anomaly_ratio"]),
                   energy_threshold=float(threshold["energy_threshold"]), elapsed_seconds=elapsed,
                   **{k: float(r[k]) for k in METRICS})
        if not math.isfinite(row["energy_threshold"]):
            raise ValueError("Non-finite actual energy threshold")
        rows.append(row)
    write_csv(output / f"{name}.csv", rows)
    best = best_row(rows, protocol["anomaly_ratios"])
    if any(not math.isfinite(best[k]) for k in METRICS):
        raise ValueError("Selected threshold has undefined metrics; retain artifacts, do not skip service")
    (output / f"{name}.best.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "service_complete", **best}, allow_nan=False), flush=True)


def summarize(output, protocol):
    rows, selected = [], []
    for service in protocol["services"]:
        with (output / f"{service['name']}.csv").open(newline="", encoding="utf-8") as handle:
            part = list(csv.DictReader(handle))
        if any(r["service"] != service["name"] for r in part):
            raise ValueError("Unexpected service in result file")
        rows.extend(part)
        chosen = best_row(part, protocol["anomaly_ratios"])
        selected.append({k: chosen[k] if k == "service" else float(chosen[k]) for k in FIELDS})
    if len(selected) != 10 or any(not math.isfinite(r[k]) for r in selected for k in METRICS):
        raise ValueError("Require all ten services with eight finite selected metrics")
    averages = {k: sum(r[k] for r in selected) / 10 for k in METRICS}
    write_csv(output / "raw.csv", rows)
    write_csv(output / "best_metrics.csv", selected)
    result = {"selection": selection_config(), "services": selected,
              "macro_average": averages, "mean_best_f1": averages["F1"]}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = [json.dumps(r, allow_nan=False) for r in selected]
    lines.append(json.dumps({"macro_average": averages}))
    lines.append('{"mean_best_f1": %.4f}' % averages["F1"])
    (output / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-service", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    cache = args.cache_dir.resolve(strict=True)
    protocol, baseline = base.configuration()
    selection = selection_config()
    if args.worker_service:
        if not args.run_validation or args.output_dir is None:
            raise ValueError("Worker requires explicit execution gate")
        service = next(s for s in protocol["services"] if s["name"] == args.worker_service)
        worker(cache, service, protocol, baseline, args.output_dir)
        return
    base.run_guards(cache)
    base.check_inputs(cache, protocol)
    if not args.run_validation:
        print(json.dumps({"f1_preflight": "ok", "selection_metric": "f_score", "metrics": METRICS}))
        return
    if args.output_dir is None:
        raise ValueError("Fresh external --output-dir required")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Results must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {"historical_data_protocol": protocol, "active_selection": selection,
                "model_parameters": baseline, "seed": 2021, "test_used": False,
                "cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()}
    (output / "run_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    for service in protocol["services"]:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--cache-dir", str(cache),
                        "--run-validation", "--output-dir", str(output),
                        "--worker-service", service["name"]], cwd=ROOT, check=True)
    base.run_guards(cache)
    summarize(output, protocol)


if __name__ == "__main__":
    main()
