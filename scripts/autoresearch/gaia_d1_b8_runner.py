"""D1 dbservice-only opt-in experiment. Default preflight; no test-set access or auto retries.

Train: --run-validation. Replay: --replay-from PREVIOUS_RESULTS (zero training).
Resume: --run-validation --resume-from PREVIOUS_RESULTS (fresh output directory).
All scoring functions and thresholds are delegated to the frozen F1 runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from scripts.autoresearch import gaia_log_v2_f1_runner as frozen

SERVICES = ["GAIA_dbservice1", "GAIA_dbservice2"]
CONFIG = ROOT / "config/autoresearch/gaia_d1_b8_contrastive.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    candidate = frozen.base.read_json(CONFIG)
    fixed = dict(services=SERVICES, seed=2021, batch_size=8, num_epochs=3,
                 gradient_train=[0, 4515], early_stopping=[4515, 5644], validation=[5644, 7056],
                 test_used=False, queue=False, clustering=False, contrastive_enabled=True,
                 llm_memory_efficient=True,
                 negative_rule="same_service_and_nonoverlapping_training_windows",
                 early_stop_monitor="fixed_rng_reconstruction_mse", selection_metric="f_score",
                 service_weighting="equal_over_exactly_two_db_instances")
    if any(candidate.get(k) != v for k, v in fixed.items()):
        raise ValueError("Candidate scope/protocol differs from approved D1 experiment")
    if not 1 <= candidate["contrast_min_negatives"] <= 7:
        raise ValueError("batch=8 has at most seven cross-modal negatives")
    full, baseline = frozen.base.configuration()
    frozen.selection_config()
    protocol = dict(full, services=[s for s in full["services"] if s["name"] in SERVICES])
    if [s["name"] for s in protocol["services"]] != SERVICES:
        raise ValueError("Missing/reordered dbservice instances")
    return candidate, protocol, baseline


def code_identity():
    # Hash source/config only; never scan raw datasets or test samples.
    paths = set()
    for directory in ("ts_benchmark", "scripts/autoresearch", "config/autoresearch"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
                paths.add(path)
    return {str(p.relative_to(ROOT)).replace(os.sep, "/"): digest(p) for p in sorted(paths)}


def resource_snapshot(output, previous=None):
    """Archive tokenizer/config needed to construct a model from full checkpoint weights."""
    destination = output / "resources" / "deepseek"
    destination.mkdir(parents=True, exist_ok=False)
    if previous:
        source = previous / "resources" / "deepseek"
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True))
    if not source.is_dir():
        raise ValueError("Missing archived model/tokenizer resources")
    hashes = {}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".tiktoken", ".py"):
            shutil.copy2(path, destination / path.name)
            hashes[path.name] = digest(path)
    if "config.json" not in hashes or not any("tokenizer" in x for x in hashes):
        raise ValueError("Incomplete tokenizer/config snapshot")
    return hashes


def make_identity(cache, candidate, protocol, objective, resources):
    import numpy as np
    from ts_benchmark.baselines.MindTS.checkpointing import environment
    logs = frozen.base.load_logs()
    prefixes = {}
    for service in protocol["services"]:
        # Only the unchanged 7056-point train/validation prefix, NEVER the test suffix.
        frame = logs.load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"],
                                         service["channels"], include_label=True)
        raw = np.ascontiguousarray(frame.to_numpy()).tobytes()
        prefixes[service["name"]] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                    "columns": list(frame.columns), "rows": len(frame)}
    return dict(source=code_identity(), config=candidate, objective=objective,
                input_prefixes=prefixes, cache_manifest=digest(cache / "manifest.json"),
                resources=resources, environment=environment())


def resource_hashes(output):
    directory = output / "resources/deepseek"
    return {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}


def service_mode(previous, name, replay=False):
    if previous is None:
        return "fresh"
    directory = previous / "checkpoints" / name
    if replay:
        if not (directory / "evaluation_start.pt").is_file():
            raise ValueError("Missing evaluation checkpoint: " + name)
        return "replay"
    # A completed service is replayed, never retrained. An interrupted one resumes.
    if (directory / "evaluation_start.pt").is_file():
        return "replay"
    if (directory / "last.pt").is_file():
        return "resume"
    if directory.exists():
        raise ValueError("Service started but has no complete checkpoint: " + name)
    return "fresh"  # Second service may never have started.


def frozen_source_guard():
    # Code/config-only subset of the historical guard. Do not hash full data files:
    # that would read the forbidden test suffix. Input prefixes are checked separately.
    full = frozen.base.read_json(ROOT / "config/autoresearch/gaia_protocol.json")
    for name, expected in full["frozen_files"].items():
        if name.startswith("dataset/"):
            continue
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)


def summarize(output, protocol):
    selected = []
    for service in protocol["services"]:
        row = frozen.base.read_json(output / (service["name"] + ".best.json"))
        if row["service"] != service["name"] or any(not math.isfinite(row[k]) for k in frozen.METRICS):
            raise ValueError("Incomplete D1 result")
        selected.append(row)
    if [r["service"] for r in selected] != SERVICES:
        raise ValueError("Both dbservice instances are mandatory; do not omit a failure")
    mean = {k: sum(row[k] for row in selected) / 2 for k in frozen.METRICS}
    frozen.write_csv(output / "best_metrics.csv", selected)
    summary = dict(scope="GAIA-D1-db-dev (two instances, project protocol, not strict paper reproduction)", services=selected,
                   macro_average=mean, mean_best_f1=mean["F1"], test_used=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [json.dumps(row) for row in selected] + ['{"mean_best_f1": %.4f}' % mean["F1"]]
    (output / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run_worker(args, cache, protocol, baseline, candidate):
    from ts_benchmark.baselines import MindTSLogged as logged_module
    from ts_benchmark.baselines.MindTSD1 import MindTSD1
    output = args.output_dir.resolve()
    saved = frozen.base.read_json(output / "run_config.json")
    current = make_identity(cache, candidate, protocol, args.objective, resource_hashes(output))
    if current != saved["identity"]:
        raise ValueError("Worker identity differs from preflight snapshot")
    name = args.worker_service
    previous = args.replay_from or args.resume_from
    mode = service_mode(previous, name, bool(args.replay_from))
    cdir = output / "checkpoints" / name
    parameters = dict(baseline, **{k: candidate[k] for k in (
        "batch_size", "num_epochs", "contrastive_enabled", "contrast_hidden", "contrast_dim",
        "contrast_dropout", "contrast_temperature", "contrast_min_negatives", "intra_weight",
        "inter_weight", "checkpoint_every", "llm_memory_efficient")})
    parameters.update(loss_mode=args.objective, checkpoint_dir=str(cdir), checkpoint_identity=current,
                      replay_resource_dir=str(output / "resources/deepseek"),
                      initialize_from_checkpoint=mode != "fresh")
    if mode == "replay":
        parameters["evaluation_checkpoint"] = str(previous / "checkpoints" / name / "evaluation_start.pt")
    if mode == "resume":
        parameters["resume_checkpoint"] = str(previous / "checkpoints" / name / "last.pt")
    # Process-local opt-in injection only. Original evaluator/threshold/scoring source stays intact.
    logged_module.MindTSLogged = MindTSD1
    service = next(s for s in protocol["services"] if s["name"] == name)
    frozen.worker(cache, service, protocol, parameters, output)
    if mode == "replay":
        # Keep this result independently replayable even if its source run is moved.
        cdir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(previous / "checkpoints" / name / "evaluation_start.pt", cdir / "evaluation_start.pt")
    if mode == "replay" and (previous / (name + ".validation_arrays.npz")).is_file():
        import numpy as np
        with np.load(previous / (name + ".validation_arrays.npz")) as old, np.load(output / (name + ".validation_arrays.npz")) as new:
            if old.files != new.files or any(not np.array_equal(old[k], new[k]) for k in old.files):
                raise RuntimeError("Exact checkpoint replay mismatch; retain both outputs for diagnosis")
        print(json.dumps({"checkpoint_replay": "exact", "service": name}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--objective", choices=("dual", "legacy_control"), default="dual")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay-from", type=Path)
    mode.add_argument("--resume-from", type=Path)
    parser.add_argument("--worker-service", choices=SERVICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    cache = args.cache_dir.resolve(strict=True)
    candidate, protocol, baseline = configure()
    frozen_source_guard()
    if args.worker_service:
        if args.output_dir is None or not (args.run_validation or args.replay_from):
            raise ValueError("Worker execution requires explicit gate")
        run_worker(args, cache, protocol, baseline, candidate)
        return
    frozen.base.check_inputs(cache, protocol)
    if not args.run_validation and not args.replay_from:
        if args.resume_from:
            raise ValueError("Resume requires --run-validation")
        print(json.dumps({"d1_preflight": "ok", "services": SERVICES, "batch_size": 8,
                          "training_started": False, "gpu_capacity_verified": False}))
        return
    if args.output_dir is None:
        raise ValueError("Fresh external output directory required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Commit code before an experiment")
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Outputs/checkpoints must be outside repository")
    # Exclusive existing experiment lock; never overlap this workflow with an old GAIA run.
    import fcntl
    with (cache.parent / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        previous = args.replay_from or args.resume_from
        if previous:
            previous = previous.resolve(strict=True)
            if args.replay_from:
                args.replay_from = previous
            else:
                args.resume_from = previous
        resources = resource_snapshot(output, previous)
        identity = make_identity(cache, candidate, protocol, args.objective, resources)
        if previous and frozen.base.read_json(previous / "run_config.json")["identity"] != identity:
            raise ValueError("Replay/resume identity mismatch (source, data, configuration or environment)")
        snapshot = dict(identity=identity, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        protocol=protocol, candidate=candidate, test_used=False, invocation=sys.argv)
        (output / "run_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        (output / "requirements-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        exit_code = 1
        try:
            for service in SERVICES:
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--cache-dir", str(cache),
                           "--output-dir", str(output), "--objective", args.objective, "--worker-service", service]
                if args.run_validation:
                    command += ["--run-validation"]
                if args.replay_from:
                    command += ["--replay-from", str(args.replay_from)]
                if args.resume_from:
                    command += ["--resume-from", str(args.resume_from)]
                subprocess.run(command, cwd=ROOT, check=True)
            if make_identity(cache, candidate, protocol, args.objective, resource_hashes(output)) != identity:
                raise RuntimeError("Source/data changed during execution")
            summarize(output, protocol)
            exit_code = 0
        finally:
            (output / "exit.status").write_text(str(exit_code) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
