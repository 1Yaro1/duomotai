"""Frozen GAIA DB baseline runner.

Default invocation performs preflight only.  Training requires the explicit
``--run-training`` gate, one B0--B4 model, seed 2021, a fresh repository-external
output directory, and a clean committed worktree.  It never opens raw GAIA CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SERVICES = ("GAIA_dbservice1", "GAIA_dbservice2")
MODEL_IDS = ("B0", "B1", "B2", "B3", "B4", "B5")
CONTRACT = ROOT / "config/autoresearch/baseline_training_contract.json"
EVALUATION_CONTRACT = ROOT / "config/evaluation_contract_db_overlap_v1.json"
TUNABLE = ROOT / "config/autoresearch/gaia_mindts_tunable.json"
OUTPUT_FIELDS = (
    "AP", "VUS-PR", "Precision", "Recall", "Point-F1", "AUC",
    "Affiliation-F", "Event Precision", "Event Recall", "Event IoU",
    ">10-minute FP count", "independent false-alarm episode count",
    "false-alarm episodes per 1000 normal points",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def code_identity():
    paths = []
    for name in (
        "ts_benchmark/baselines/MindTS/MindTS.py",
        "ts_benchmark/baselines/MindTS/models/MindTS_model.py",
        "ts_benchmark/baselines/MindTS/cmc_log_data.py",
        "ts_benchmark/baselines/MindTS/deterministic_scoring.py",
        "ts_benchmark/baselines/MindTSWeb.py",
        "ts_benchmark/baselines/MindTSDBBaselines.py",
        "scripts/autoresearch/gaia_db_baseline_runner.py",
        "config/autoresearch/baseline_training_contract.json",
        "config/evaluation_contract_db_overlap_v1.json",
    ):
        path = ROOT / name
        paths.append((name, digest(path)))
    return dict(paths)


def validate_contract(prefix_dir, log_cache):
    contract = read_json(CONTRACT)
    if contract["status"] != "pretraining_frozen" or contract["training_started"] is not False:
        raise ValueError("Baseline contract is not in the pretraining-frozen state")
    if contract["services"] != list(SERVICES):
        raise ValueError("Service list/order changed")
    if contract["splits"] != {
        "gradient_train": [0, 4515], "early_stop": [4515, 5644],
        "proposal_validation": [5644, 7056], "final_test": "forbidden"}:
        raise ValueError("Split contract changed")
    expected_eval = contract["evaluation"]["contract_sha256"]
    if digest(EVALUATION_CONTRACT) != expected_eval:
        raise ValueError("Frozen evaluation contract changed")
    evaluation = read_json(EVALUATION_CONTRACT)
    required_completion = (
        "historical_metrics_verified", "deterministic_overlap_v1_tests_passed",
        "event_diagnosis_complete", "evaluation_contract_frozen",
        "ready_for_unimodal_baselines")
    if not all(evaluation["completion"].get(key) is True for key in required_completion):
        raise ValueError("Frozen evaluation prerequisites are incomplete")
    if evaluation["execution_policy"]["test_used"] is not False:
        raise ValueError("Evaluation contract reports test use")
    if digest(log_cache / "manifest.json") != contract["input_policy"]["log_v2_manifest_sha256"]:
        raise ValueError("Log-v2 manifest changed")
    manifest = read_json(prefix_dir / "prefix_manifest.json")
    if any(manifest.get(key) is not False for key in (
            "test_values_parsed", "test_values_materialized", "test_values_saved",
            "test_values_used", "test_labels_used")):
        raise ValueError("Prefix-only provenance violates the no-test contract")
    import numpy as np
    for service in SERVICES:
        record = contract["input_policy"]["prefix_cache_files"][service]
        path = prefix_dir / record["file"]
        if digest(path) != record["file_sha256"]:
            raise ValueError("Prefix cache changed: " + service)
        with np.load(path, allow_pickle=False) as data:
            values = data["values"]
            columns = data["columns"]
            absolute = data["absolute_index"]
            if (values.shape != (7056, record["metric_channels"] + 1)
                    or columns.shape != (record["metric_channels"] + 1,)
                    or columns[-1] != "label"
                    or not np.array_equal(absolute, np.arange(7056))):
                raise ValueError("Prefix shape/order mismatch: " + service)
            canonical = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            if canonical != record["canonical_array_sha256"]:
                raise ValueError("Canonical prefix identity mismatch: " + service)
    return contract


def resource_snapshot(output):
    from huggingface_hub import snapshot_download
    source = Path(snapshot_download(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True))
    destination = output / "resources/deepseek"
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".tiktoken", ".py"):
            shutil.copy2(path, destination / path.name)
            hashes[path.name] = digest(path)
    return hashes


def intervals(binary):
    import numpy as np
    values = np.asarray(binary, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    return [(int(start), int(end - 1)) for start, end in
            zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def event_diagnostics(actual, predicted):
    import numpy as np
    truth, estimate = intervals(actual), intervals(predicted)
    overlap = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    hit_pred = [p for p in estimate if any(overlap(p, t) for t in truth)]
    hit_truth = [t for t in truth if any(overlap(p, t) for p in estimate)]
    ious = []
    for true in truth:
        candidates = [overlap(true, pred) / (
            true[1] - true[0] + 1 + pred[1] - pred[0] + 1 - overlap(true, pred))
                      for pred in estimate if overlap(true, pred)]
        ious.append(max(candidates, default=0.0))
    independent = [p for p in estimate if not any(overlap(p, t) for t in truth)]
    positive = np.flatnonzero(actual)
    fp_index = np.flatnonzero(np.asarray(predicted, bool) & ~np.asarray(actual, bool))
    distances = (np.min(np.abs(fp_index[:, None] - positive[None, :]), axis=1)
                 if len(fp_index) else np.empty(0, dtype=np.int64))
    normal_points = int((np.asarray(actual) == 0).sum())
    return {
        "Event Precision": len(hit_pred) / len(estimate) if estimate else 0.0,
        "Event Recall": len(hit_truth) / len(truth) if truth else 0.0,
        "Event IoU": float(np.mean(ious)) if ious else 0.0,
        ">10-minute FP count": int((distances > 10).sum()),
        "independent false-alarm episode count": len(independent),
        "false-alarm episodes per 1000 normal points": (
            len(independent) * 1000.0 / normal_points if normal_points else math.nan),
        "true_event_count": len(truth),
        "predicted_event_count": len(estimate),
    }


def evaluate(service, actual, score, channel_error, ratios, thresholds, predictions, columns):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ts_benchmark.evaluation.metrics import classification_metrics_label as metrics
    rows = []
    for ratio, threshold, pred in zip(ratios, thresholds, predictions):
        pred = pred.astype(bool)
        tp = int(((actual == 1) & pred).sum())
        fp = int(((actual == 0) & pred).sum())
        fn = int(((actual == 1) & ~pred).sum())
        tn = int(((actual == 0) & ~pred).sum())
        rows.append({"anomaly_ratio": float(ratio), "threshold": float(threshold),
                     "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "Precision": tp / (tp + fp) if tp + fp else 0.0,
                     "Recall": tp / (tp + fn) if tp + fn else 0.0,
                     "Point-F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
    chosen = max(rows, key=lambda row: row["Point-F1"])
    index = rows.index(chosen)
    pred = predictions[index].astype(bool)
    chosen.update({
        "service": service,
        "AP": float(average_precision_score(actual, score)),
        "AUC": float(roc_auc_score(actual, score)),
        "Affiliation-F": float(metrics.affiliation_f(actual, pred, score)),
        "VUS-PR": float(metrics.VUS_PR(actual, pred, score)),
        "VUS-ROC": float(metrics.VUS_ROC(actual, pred, score)),
    })
    chosen.update(event_diagnostics(actual, pred))
    fp = pred & (actual == 0)
    if fp.any():
        top = np.argmax(channel_error[fp], axis=1)
        unique, count = np.unique(top, return_counts=True)
        order = np.argsort(-count, kind="stable")
        recurring = [{"channel": str(columns[unique[i]]), "fp_top1_count": int(count[i])}
                     for i in order[:10]]
    else:
        recurring = []
    chosen["recurring top-error channels"] = recurring
    return chosen, rows


def model_parameters(contract, model_id, service, output, identity, log_cache):
    baseline = read_json(TUNABLE)
    spec = {
        "B0": ("metric_only_pure", "unused"),
        "B1": ("metric_only_stats", "unused"),
        "B2": ("metric_only_null_log", "null"),
        "B3": ("metric_log_shifted", "shifted"),
        "B4": ("metric_log_simple", "aligned"),
    }[model_id]
    baseline.update({
        "seed": 2021, "batch_size": 8, "num_epochs": 3,
        "parallel_strategy": None, "log_input_mode": "v2",
        "defer_log_store": True, "log_v2_cache_dir": str(log_cache),
        "log_v2_service": service, "log_semantic_dim": 768,
        "log_count_dim": 12, "log_shift_minutes": 240,
        "enc_in_time": contract["input_policy"]["prefix_cache_files"][service]["metric_channels"],
        "baseline_id": model_id, "baseline_mode": spec[0],
        "baseline_log_source": spec[1], "contrastive_enabled": False,
        "loss_mode": "reconstruction_ib", "llm_memory_efficient": True,
        "checkpoint_every": 50, "max_optimizer_updates": 1686,
        "checkpoint_dir": str(output / "checkpoints" / service),
        "checkpoint_identity": identity,
        "replay_resource_dir": str(output / "resources/deepseek"),
        "initialize_from_checkpoint": False,
    })
    return baseline


def run_worker(args, contract):
    import numpy as np
    import pandas as pd
    import torch
    from ts_benchmark.baselines.MindTSDBBaselines import MindTSDBBaseline
    from ts_benchmark.baselines.MindTS.deterministic_scoring import score_threshold_and_prediction
    from ts_benchmark.utils.random_utils import fix_random_seed

    output = args.output_dir.resolve()
    snapshot = read_json(output / "run_config.json")
    if snapshot["identity"]["source"] != code_identity():
        raise ValueError("Source changed after preflight")
    service = args.worker_service
    record = contract["input_policy"]["prefix_cache_files"][service]
    with np.load(args.prefix_dir / record["file"], allow_pickle=False) as data:
        values = data["values"].copy()
        columns = data["columns"].astype(str)
    metrics, labels = values[:, :-1], values[:, -1].astype(np.int8)
    if not np.array_equal(labels, labels.astype(bool)):
        raise ValueError("Labels are not binary")
    train_frame = pd.DataFrame(metrics[:5644], index=np.arange(5644), columns=columns[:-1])
    parameters = model_parameters(
        contract, args.model, service, output, snapshot["identity"], args.log_cache_dir)
    fix_random_seed(2021)
    trainer = MindTSDBBaseline(**parameters)
    started = time.monotonic()
    trainer.detect_multi_fit(train_frame, None, None)
    if trainer.progress["global_step"] > 1686:
        raise RuntimeError("Optimizer update contract exceeded")
    trainer._save("evaluation_start")
    trainer.model.eval()
    scaled = trainer.scaler.transform(metrics)
    training = trainer.scoring_dataset(scaled[:4515], 0)
    validation = trainer.scoring_dataset(scaled[5644:7056], 5644)
    passes = []
    torch.cuda.reset_peak_memory_stats()
    for pass_id in (1, 2):
        if pass_id == 2:
            torch.rand(113, device=trainer.device)
            np.random.rand(113)
        with patch("torch.rand", side_effect=AssertionError("Eval torch.rand is forbidden")), patch(
                "torch.nn.functional.gumbel_softmax",
                side_effect=AssertionError("Eval Gumbel is forbidden")):
            train_result, val_result, ratios, thresholds, predicted = score_threshold_and_prediction(
                trainer.model, training, validation, device=trainer.device)
        flat = {"train_" + key: value for key, value in train_result.items()}
        flat.update({"validation_" + key: value for key, value in val_result.items()})
        flat.update(anomaly_ratio=ratios, thresholds=thresholds,
                    predicted=predicted, actual=labels[5644:7056])
        with (output / service / f"pass{pass_id}.npz").open("xb") as handle:
            np.savez_compressed(handle, **flat)
        passes.append(flat)
    for key in passes[0]:
        if (passes[0][key].dtype != passes[1][key].dtype
                or passes[0][key].shape != passes[1][key].shape
                or passes[0][key].tobytes() != passes[1][key].tobytes()):
            raise AssertionError("Nondeterministic baseline score array: " + key)
    flat = passes[0]
    selected, candidates = evaluate(
        service, flat["actual"], flat["validation_score"],
        flat["validation_channel_error"], flat["anomaly_ratio"],
        flat["thresholds"], flat["predicted"], columns[:-1])
    result = {
        "model_id": args.model, "service": service,
        "score_protocol": "deterministic_overlap_v1", "metrics": selected,
        "threshold_candidates": candidates, "two_passes_bitwise_equal": True,
        "optimizer_updates": trainer.progress["global_step"],
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "test_used": False,
    }
    (output / service / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"event": "service_complete", "service": service,
                      "model_id": args.model, "metrics": selected}, allow_nan=False), flush=True)


def summarize(output, model_id):
    results = [read_json(output / service / "result.json") for service in SERVICES]
    if [row["service"] for row in results] != list(SERVICES):
        raise ValueError("Both DB services are mandatory")
    selected = [row["metrics"] for row in results]
    macro = {key: sum(float(row[key]) for row in selected) / len(selected)
             for key in OUTPUT_FIELDS}
    summary = {"model_id": model_id, "services": selected, "macro_average": macro,
               "all_services_successful": True, "test_used": False}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "all_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        scalar_fields = ["model_id", "service", *OUTPUT_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({"model_id": model_id, "service": row["service"],
                             **{key: row[key] for key in OUTPUT_FIELDS}})
        writer.writerow({"model_id": model_id, "service": "macro_average", **macro})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--log-cache-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--worker-service", choices=SERVICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.prefix_dir = args.prefix_dir.resolve(strict=True)
    args.log_cache_dir = args.log_cache_dir.resolve(strict=True)
    contract = validate_contract(args.prefix_dir, args.log_cache_dir)
    if args.worker_service:
        if not args.run_training or args.model not in MODEL_IDS[:5] or args.output_dir is None:
            raise ValueError("Worker requires an explicit B0--B4 training gate")
        run_worker(args, contract)
        return
    if not args.run_training:
        print(json.dumps({"baseline_preflight": "ok", "models": MODEL_IDS,
                          "seed": 2021, "training_started": False,
                          "test_used": False, "contract_sha256": digest(CONTRACT)}))
        return
    if args.model is None or args.output_dir is None:
        raise ValueError("Training requires --model and a fresh --output-dir")
    if args.model == "B5":
        raise ValueError("B5 is a frozen historical reference and must never be retrained")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Commit and audit the baseline code before training")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("All run outputs must be outside the Git repository")
    output.mkdir(parents=True, exist_ok=False)
    exit_code = 1
    try:
        resources = resource_snapshot(output)
        expected_resources = contract["architecture"]["qwen"]["resource_hashes"]
        if any(resources.get(name) != expected for name, expected in expected_resources.items()):
            raise ValueError("Frozen Qwen tokenizer/config resources changed")
        identity = {
            "source": code_identity(), "contract_sha256": digest(CONTRACT),
            "evaluation_contract_sha256": digest(EVALUATION_CONTRACT),
            "model_id": args.model, "seed": 2021,
            "prefix_manifest_sha256": digest(args.prefix_dir / "prefix_manifest.json"),
            "log_manifest_sha256": digest(args.log_cache_dir / "manifest.json"),
            "resources": resources,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        (output / "run_config.json").write_text(json.dumps({
            "identity": identity, "contract": contract, "invocation": sys.argv,
            "test_used": False}, indent=2), encoding="utf-8")
        for service in SERVICES:
            (output / service).mkdir()
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--prefix-dir", str(args.prefix_dir), "--log-cache-dir", str(args.log_cache_dir),
                "--model", args.model, "--output-dir", str(output),
                "--run-training", "--worker-service", service,
            ]
            subprocess.run(command, cwd=ROOT, check=True)
        if code_identity() != identity["source"]:
            raise RuntimeError("Source changed during the experiment")
        summarize(output, args.model)
        exit_code = 0
    finally:
        (output / "exit.status").write_text(str(exit_code) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
