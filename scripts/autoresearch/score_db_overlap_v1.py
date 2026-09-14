"""Existing best.pt only: identity audit or two full deterministic score passes.

Historical predictions are reused after best/evaluation_start weight identity
verification, never replayed. No optimizer construction, training or raw CSV.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ts_benchmark.baselines.MindTS.checkpointing import digest, environment
from ts_benchmark.baselines.MindTS.deterministic_scoring import (
    AbsoluteWindowDataset, score_threshold_and_prediction)


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def tensor_identity(first, second):
    assert first.keys() == second.keys(), "Model state keys differ"
    for name in first:
        a, b = first[name], second[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() == b.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--exact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", choices=["GAIA_dbservice1", "GAIA_dbservice2"], required=True)
    parser.add_argument("--execute-score", action="store_true", help="Default: CPU identity checks only")
    parser.add_argument("--accept-historical-prefix-hash", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Results must be outside Git")
    args.output.mkdir(parents=True, exist_ok=False)
    exact = json.loads(args.exact_report.read_text())
    assert exact["exact_replay"] is True
    assert all(x["bitwise_equal"] for x in exact["checks"])
    run = json.loads((args.run / "run_config.json").read_text())
    identity = run["identity"]
    prefix_manifest = json.loads((args.prefix / "prefix_manifest.json").read_text())
    attestation = json.loads((args.prefix / "prefix_identity_attestation.json").read_text())
    assert attestation["historical_prefix_elementwise_comparison"] is False
    for field in ("historical_prefix_identity_verified_by_sha256", "prefix_shape_verified",
                  "prefix_column_order_verified", "validation_labels_elementwise_verified"):
        assert attestation[field] is True
    record = prefix_manifest["services"][args.service]
    for field in ("test_values_parsed", "test_values_materialized", "test_values_saved",
                  "test_values_used", "test_labels_used"):
        assert prefix_manifest[field] is False
    prefix_file = args.prefix / (args.service + ".prefix.npz")
    assert digest(prefix_file) == record["cache_sha256"]
    with np.load(prefix_file, allow_pickle=False) as cache:
        values, columns, indices = [cache[k].copy() for k in ("values", "columns", "absolute_index")]
    reference = identity["input_prefixes"][args.service]
    assert hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == reference["sha256"]
    assert attestation["services"][args.service]["historical_prefix_sha256"] == reference["sha256"]
    assert attestation["services"][args.service]["current_canonical_prefix_sha256"] == reference["sha256"]
    assert attestation["services"][args.service]["cache_file_sha256"] == digest(prefix_file)
    assert values.shape == (7056, len(reference["columns"]))
    assert columns.tolist() == reference["columns"]
    assert np.array_equal(indices, np.arange(7056))
    cpdir = args.run / "checkpoints" / args.service
    best = torch.load(cpdir / "best.pt", map_location="cpu", weights_only=False, mmap=True)
    evaluation = torch.load(cpdir / "evaluation_start.pt", map_location="cpu", weights_only=False, mmap=True)
    assert best["kind"] == "best" and evaluation["kind"] == "evaluation_start"
    assert best["identity"] == evaluation["identity"] == identity
    tensor_identity(best["model"], evaluation["model"])
    assert best["scaler"].keys() == evaluation["scaler"].keys()
    for key in best["scaler"]:
        a, b = best["scaler"][key], evaluation["scaler"][key]
        assert np.array_equal(a, b), ("scaler", key)
    del evaluation
    resources = args.run / "resources/deepseek"
    for name, expected in identity["resources"].items():
        assert digest(resources / name) == expected
    original_arrays = args.run / (args.service + ".validation_arrays.npz")
    with np.load(original_arrays, allow_pickle=False) as historical:
        actual = historical["actual"].copy()
    assert np.array_equal(actual, values[5644:, -1])
    provenance = dict(service=args.service, checkpoint=str(cpdir / "best.pt"),
        best_weights_byte_equal_evaluation_start=True, scaler_equal=True,
        historical_score_source="reuse already byte-verified arrays; NO historical forward/replay",
        historical_arrays_sha256=digest(original_arrays), prefix_manifest=prefix_manifest,
        prefix_identity_attestation=attestation,
        exact_replay_repeated=False, training_updates=0, test_used=False,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={name: digest(ROOT / name) for name in (
            "scripts/autoresearch/score_db_overlap_v1.py",
            "ts_benchmark/baselines/MindTS/deterministic_scoring.py",
            "ts_benchmark/baselines/MindTS/models/MindTS_model.py")})
    write_json(args.output / "identity.json", provenance)
    if not args.execute_score:
        print(json.dumps(dict(identity_verified=True, scored=False, training_updates=0)), flush=True)
        return
    if record["historical_full_prefix_direct_elementwise_comparison"] is not True and not args.accept_historical_prefix_hash:
        raise ValueError("Historical full prefix unavailable: explicit acceptance of SHA256 identity is required")
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible GPU required")
    free_mib = torch.cuda.mem_get_info()[0] // 1024**2
    if free_mib < 21000:
        raise RuntimeError(f"GPU free {free_mib} MiB < fixed 21000 MiB admission requirement; no retry")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    env = environment()
    if env != identity["environment"]:
        raise ValueError("Environment differs from audited run; no automatic parameter changes")
    from ts_benchmark.baselines.MindTS.cmc_log_data import LogV2Store
    from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
    config = SimpleNamespace(**best["config"])
    config.initialize_from_checkpoint = True
    config.replay_resource_dir = str(resources)
    model = MINDTSModel(config)
    model.load_state_dict(best["model"], strict=True)
    model.to("cuda").eval()
    model.requires_grad_(False)
    scaler = StandardScaler()
    scaler.__dict__.update(best["scaler"])
    del best
    gc.collect()
    import pandas as pd
    standardized = scaler.transform(pd.DataFrame(values[:, :-1], columns=columns[:-1]))
    logs = LogV2Store(config.log_v2_cache_dir, args.service)
    train = AbsoluteWindowDataset(standardized[:4515], 0, logs)
    validation = AbsoluteWindowDataset(standardized[5644:7056], 5644, logs)
    all_passes = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for iteration in (1, 2):
        def progress(done, total):
            if done % 80 == 0 or done == total:
                print(json.dumps(dict(service=args.service, pass_id=iteration, windows=done,
                                      total_windows=total, elapsed_seconds=time.monotonic()-started)), flush=True)
        # Change global RNG between passes, not reset to a convenient seed.
        if iteration == 2:
            torch.rand(113, device="cuda")
            np.random.rand(113)
        with patch("torch.rand", side_effect=AssertionError("Eval torch.rand is forbidden")), patch(
                "torch.nn.functional.gumbel_softmax", side_effect=AssertionError("Eval Gumbel is forbidden")):
            result = score_threshold_and_prediction(model, train, validation, device="cuda", progress=progress)
        training_result, val_result, ratios, thresholds, predictions = result
        flat = {"train_"+k: v for k, v in training_result.items()}
        flat.update({"validation_"+k: v for k, v in val_result.items()})
        flat.update(anomaly_ratio=ratios, thresholds=thresholds, predicted=predictions, actual=actual)
        with (args.output / f"pass{iteration}.npz").open("xb") as f:
            np.savez_compressed(f, **flat)
        all_passes.append(flat)
    for key in all_passes[0]:
        a, b = all_passes[0][key], all_passes[1][key]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            mismatch = np.flatnonzero(a.ravel() != b.ravel())
            raise AssertionError(f"Nondeterministic {key}; first unequal index {mismatch[:1].tolist()}")
    # Independent standard metrics; original affiliation/VUS implementation.
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ts_benchmark.evaluation.metrics import classification_metrics_label as metrics
    rows = []
    for i, (ratio, threshold) in enumerate(zip(ratios, thresholds)):
        pred = predictions[i]
        tp, fp = int(((actual == 1) & pred).sum()), int(((actual == 0) & pred).sum())
        fn, tn = int(((actual == 1) & ~pred).sum()), int(((actual == 0) & ~pred).sum())
        rows.append(dict(anomaly_ratio=float(ratio), threshold=float(threshold), TP=tp, FP=fp, TN=tn, FN=fn,
                         PR=tp/(tp+fp) if tp+fp else 0., RC=tp/(tp+fn), F1=2*tp/(2*tp+fp+fn)))
    chosen = max(rows, key=lambda row: row["F1"])
    best_index = rows.index(chosen)
    scores = val_result["score"]
    chosen.update(AP=float(average_precision_score(actual, scores)), AUC=float(roc_auc_score(actual, scores)))
    for key, function in (("affiliation_f", "affiliation_f"), ("V-ROC", "VUS_ROC"), ("V-PR", "VUS_PR")):
        chosen[key] = float(getattr(metrics, function)(actual, predictions[best_index], scores))
    write_json(args.output / "result.json", dict(service=args.service, score_protocol="deterministic_overlap_v1",
        best=chosen, all_candidates=rows, two_full_passes_bitwise_equal=True,
        validation_points=1412, all_points_covered=True, tail_padding=False,
        environment=env, elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), training_updates=0, test_used=False))
    print(json.dumps(dict(service=args.service, complete=True, best=chosen)), flush=True)


if __name__ == "__main__":
    main()
