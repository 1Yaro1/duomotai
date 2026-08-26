#!/usr/bin/env python3
"""Fail if frozen GAIA data, evaluation code, or protocol invariants drift."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_TUNABLE_KEYS = {
    "activation",
    "batch_size",
    "d_ff",
    "d_model",
    "dropout",
    "e_layers",
    "factor",
    "horizon",
    "lamda",
    "lamda1",
    "lamda2",
    "lr",
    "mask_ratio",
    "n_heads",
    "norm",
    "num_epochs",
    "parallel_strategy",
    "patch_size",
    "patience",
    "r",
    "seq_len",
    "stride",
    "use_norm",
}
EXPECTED_SPLIT = {
    "fit_start": 0,
    "fit_end": 5644,
    "validation_start": 5644,
    "validation_end": 7056,
    "test_start": 7056,
    "test_end": 10080,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_python(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> int:
    protocol_path = PROJECT_ROOT / "config/autoresearch/gaia_protocol.json"
    tunable_path = PROJECT_ROOT / "config/autoresearch/gaia_mindts_tunable.json"
    protocol = load_json(protocol_path)
    tunable = load_json(tunable_path)

    require(protocol.get("schema_version") == 1, "unsupported protocol schema")
    require(protocol.get("seed") == 2021, "seed drift")
    require(protocol.get("metric") == "affiliation_f", "metric drift")
    require(protocol.get("split") == EXPECTED_SPLIT, "split drift")
    ratios = [float(value) for value in protocol.get("anomaly_ratios", [])]
    require(len(ratios) == 61 and len(set(ratios)) == 61, "threshold grid drift")
    services = protocol.get("services", [])
    require(len(services) == 10, "service count drift")
    require(len({entry["name"] for entry in services}) == 10, "duplicate services")

    extra_tunable = set(tunable) - ALLOWED_TUNABLE_KEYS
    require(not extra_tunable, f"unsupported tunable keys: {sorted(extra_tunable)}")
    for key, value in tunable.items():
        require(not isinstance(value, float) or math.isfinite(value), f"non-finite {key}")

    for relative, expected_hash in protocol["frozen_files"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"missing frozen file: {relative}")
        require(sha256(path) == expected_hash, f"frozen file changed: {relative}")

    manifest_path = PROJECT_ROOT / "dataset/anomaly_detect/GAIA_instances/manifest.json"
    manifest = load_json(manifest_path)
    quality = {entry["service"]: entry for entry in manifest["quality_summary"]}
    metadata_path = PROJECT_ROOT / "dataset/anomaly_detect/DETECT_META.csv"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata = {row["file_name"]: row for row in csv.DictReader(handle)}

    for entry in services:
        short_name = entry["name"].removeprefix("GAIA_")
        require(short_name in quality, f"missing manifest service: {short_name}")
        item = quality[short_name]
        require(int(item["time_steps"]) == 10080, f"length drift: {entry['name']}")
        require(int(item["train_length"]) == 7056, f"outer split drift: {entry['name']}")
        require(int(item["metric_channels"]) == int(entry["channels"]), f"channel drift: {entry['name']}")
        series_path = PROJECT_ROOT / "dataset/anomaly_detect/data" / entry["series"]
        text_path = PROJECT_ROOT / "dataset/anomaly_detect/data" / entry["text"]
        require(sha256(series_path) == item["metric_sha256"], f"series hash drift: {entry['name']}")
        require(sha256(text_path) == item["text_sha256"], f"text hash drift: {entry['name']}")
        meta = metadata.get(entry["series"])
        require(meta is not None, f"missing metadata row: {entry['series']}")
        require(int(meta["train_lens"]) == 7056, f"metadata train split drift: {entry['name']}")
        require(int(meta["time_steps"]) == 10080, f"metadata length drift: {entry['name']}")
        require(int(meta["total_len"]) == 10080, f"metadata total drift: {entry['name']}")

    for relative in [
        "scripts/autoresearch/gaia_validation_runner.py",
        "scripts/autoresearch/summarize_gaia_best_f1.py",
        "scripts/autoresearch/guard_gaia_protocol.py",
        "ts_benchmark/baselines/MindTS/MindTS.py",
        "ts_benchmark/baselines/MindTS/models/MindTS_model.py",
        "ts_benchmark/baselines/MindTS/layers/Embed.py",
        "ts_benchmark/baselines/MindTS/layers/Transformer_EncDec.py",
        "ts_benchmark/baselines/MindTS/layers/SelfAttention_Family.py",
    ]:
        parse_python(PROJECT_ROOT / relative)

    print('{"guard":"ok"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
