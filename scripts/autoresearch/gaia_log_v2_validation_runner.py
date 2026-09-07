"""Log-only GAIA ablation entrypoint. Default: checks only, never training.

--run-validation is an explicit execution gate. It keeps the old losses,
anomaly scoring, 61 thresholds, evaluation strategy and read-only summarizer.
Every service runs in a fresh child process; all ten are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FIELDS = ["service", "series", "text", "channels", "anomaly_ratio", "affiliation_f", "elapsed_seconds", "log_info"]


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def load_logs():
    path = ROOT / "ts_benchmark/baselines/MindTS/cmc_log_data.py"
    spec = importlib.util.spec_from_file_location("log_v2_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_guards(cache):
    for script, extra in (("guard_gaia_protocol.py", []), ("guard_gaia_log_v2.py", ["--cache-dir", str(cache)])):
        subprocess.run([sys.executable, str(ROOT / "scripts/autoresearch" / script), *extra], cwd=ROOT, check=True)


def configuration():
    protocol = read_json(ROOT / "config/autoresearch/gaia_protocol.json")
    candidate = read_json(ROOT / "config/autoresearch/gaia_log_v2.json")
    baseline_path = ROOT / "config/autoresearch/gaia_mindts_tunable.json"
    baseline = read_json(baseline_path)
    if candidate.get("baseline_model_config_sha256") != hashlib.sha256(baseline_path.read_bytes()).hexdigest():
        raise ValueError("Baseline hyperparameters changed: this must remain a log-only ablation")
    if candidate.get("cmc_loss_enabled") is not False or candidate.get("seed") != 2021:
        raise ValueError("CMC must remain disabled and seed fixed")
    frozen = {"gradient_train": [0, 4515], "early_stopping": [4515, 5644],
              "proposal_validation": [5644, 7056], "count_scaler_fit_interval": [0, 4515],
              "window": 24, "generate_test_features": False,
              "semantic_encoder_frozen": True,
              "loss_and_anomaly_score": "unchanged_reconstruction_baseline"}
    if any(candidate.get(key) != value for key, value in frozen.items()):
        raise ValueError("Candidate declaration differs from the fixed log-only protocol")
    if candidate.get("services") != [s["name"] for s in protocol["services"]]:
        raise ValueError("Candidate service order/list differs from the frozen protocol")
    return protocol, baseline


def check_inputs(cache, protocol):
    logs = load_logs()
    reports = []
    for service in protocol["services"]:
        store = logs.LogV2Store(cache, service["name"])
        # No labels are loaded during this diagnostic.
        frame = logs.load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"], int(service["channels"]))
        frame = frame.reset_index(drop=True)  # Same absolute row IDs as frozen strategy.
        windows = {}
        for name, lo, hi in (("train", 0, 4515), ("val", 4515, 5644),
                             ("test", 5644, 7056), ("thre", 5644, 7056)):
            ds = logs.LogV2Dataset(frame.iloc[lo:hi], store, mode=name)
            for index in (0, len(ds) - 1):
                numeric, feature, present, _ = ds[index]
                if numeric.shape != (24, int(service["channels"])) or feature.shape[0] != 24 or present.shape != (24,):
                    raise ValueError("Bad aligned window shape")
            windows[name] = len(ds)
        report = {"service": service["name"], "rows": len(frame), "windows": windows,
                  "events": int(store.counts.sum()), "templates": store.count_dim - 1,
                  "unknown_events": int(store.counts[:, 0].sum()),
                  "missing_minutes": int((~store.present).sum()), "semantic_dim": store.semantic_dim,
                  "alignment": "absolute_zero_based_minute_ids", "labels_read": False}
        reports.append(report)
        print(json.dumps(report), flush=True)
    print(json.dumps({"log_v2_check": "ok", "services": len(reports), "training_started": False}), flush=True)
    return reports


def worker(cache, service, protocol, baseline, output):
    import numpy as np
    import pandas as pd
    from ts_benchmark.baselines.MindTSLogged import MindTSLogged
    from ts_benchmark.data.data_pool import DataPool
    from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
    from ts_benchmark.data.dataset import Dataset
    from ts_benchmark.evaluation.evaluator import Evaluator
    from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
    from ts_benchmark.models import ModelFactory
    from ts_benchmark.utils.random_utils import fix_random_seed

    logs = load_logs()
    series = logs.load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"],
                                      int(service["channels"]), include_label=True)
    # v2 reads the sidecar; this shape-only placeholder satisfies the unchanged
    # strategy signature and never enters the log encoder.
    text = pd.DataFrame({"channel1": [""] * len(series)}, index=series.index)
    source = LocalAnomalyDetectDataSource()  # Metadata only; no load_series_list.
    metadata = source.dataset.metadata.loc[[service["series"]]].copy(deep=True)
    for key, value in {"train_lens": 5644, "time_steps": 7056, "total_len": 7056, "train/total": 5644 / 7056}.items():
        metadata.loc[service["series"], key] = value
    pool = Dataset()
    pool.set_data(data_dict={service["series"]: series}, covariate_dict={}, metadata=metadata)
    pool.update_multi_data({}, {service["text"]: text}, {})
    DataPool().set_pool(pool)
    params = dict(baseline, log_input_mode="v2", log_v2_cache_dir=str(cache),
                  log_v2_service=service["name"], enc_in_time=int(service["channels"]),
                  anomaly_ratio=list(protocol["anomaly_ratios"]))
    fix_random_seed(2021)
    strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label"},
                                  Evaluator([{"name": protocol["metric"]}]))
    factory = ModelFactory("MindTSLogged", MindTSLogged, params)
    started = time.monotonic()
    raw = strategy.multi_execute(service["series"], service["text"], factory)
    elapsed = time.monotonic() - started
    rows = []
    for record in raw:
        row = dict(zip(strategy.field_names, record))
        rows.append(dict(service=service["name"], series=service["series"], text=service["text"],
                         channels=service["channels"], anomaly_ratio=row.get("typical_anomaly_ratio"),
                         affiliation_f=row.get("affiliation_f"), elapsed_seconds=elapsed,
                         log_info=row.get("log_info", "")))
    valid = [r for r in rows if isinstance(r["affiliation_f"], (int, float, np.floating)) and math.isfinite(float(r["affiliation_f"]))]
    if len(rows) != 61 or not valid:
        raise RuntimeError("Incomplete service result: " + repr(rows))
    with output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    best = max(valid, key=lambda r: float(r["affiliation_f"]))
    print(json.dumps({"service": service["name"], "best_f1": float(best["affiliation_f"]), "elapsed_seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-validation", action="store_true", help="Explicitly permit complete ten-service training/validation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-service", choices=load_logs().SERVICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    cache = args.cache_dir.resolve(strict=True)
    protocol, baseline = configuration()
    if args.worker_service:
        if not args.run_validation or args.output_dir is None:
            raise ValueError("Worker requires explicit execution gate and output")
        service = next(s for s in protocol["services"] if s["name"] == args.worker_service)
        worker(cache, service, protocol, baseline, args.output_dir / f"{args.worker_service}.csv")
        return
    run_guards(cache)
    check_inputs(cache, protocol)
    if not args.run_validation:
        return
    if args.output_dir is None:
        raise ValueError("--run-validation requires a fresh --output-dir")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Results must be outside the Git repository")
    output.mkdir(parents=True, exist_ok=False)
    write = {"protocol": protocol, "model_parameters": baseline, "seed": 2021,
             "log_input_mode": "v2", "cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()}
    (output / "run_config.json").write_text(json.dumps(write, indent=2), encoding="utf-8")
    for service in protocol["services"]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cache-dir", str(cache),
                        "--run-validation", "--output-dir", str(output),
                        "--worker-service", service["name"]], cwd=ROOT, check=True)
    raw = output / "raw.csv"
    with raw.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for service in protocol["services"]:
            with (output / f"{service['name']}.csv").open(newline="", encoding="utf-8") as part:
                writer.writerows(csv.DictReader(part))
    run_guards(cache)
    # Last nonempty line is the original summarizer's {"mean_best_f1": 0.0000}.
    subprocess.run([sys.executable, str(ROOT / "scripts/autoresearch/summarize_gaia_best_f1.py"),
                    "--protocol", str(ROOT / "config/autoresearch/gaia_protocol.json"),
                    "--input", str(raw)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
