#!/usr/bin/env python3
"""Run the frozen GAIA inner-validation protocol without touching the test split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from ts_benchmark.baselines.MindTSLogged import MindTSLogged
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.data.dataset import Dataset
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.random_utils import fix_random_seed


PROTOCOL_KEYS = {
    "schema_version",
    "seed",
    "metric",
    "split",
    "anomaly_ratios",
    "services",
    "frozen_files",
}
FORBIDDEN_TUNABLE_KEYS = {
    "anomaly_ratio",
    "anomaly_ratios",
    "enc_in",
    "enc_in_time",
    "metric",
    "seed",
    "services",
    "split",
    "test_start",
    "test_end",
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


def ensure_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("raw validation output must be outside the Git repository")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_configuration(protocol: dict[str, Any], tunable: dict[str, Any]) -> None:
    if set(protocol) != PROTOCOL_KEYS:
        raise ValueError(
            f"unexpected protocol keys: missing={sorted(PROTOCOL_KEYS - set(protocol))}, "
            f"extra={sorted(set(protocol) - PROTOCOL_KEYS)}"
        )
    if protocol["schema_version"] != 1:
        raise ValueError("unsupported protocol schema")
    if protocol["seed"] != 2021:
        raise ValueError("GAIA validation seed must remain 2021")
    if protocol["metric"] != "affiliation_f":
        raise ValueError("GAIA validation metric must remain affiliation_f")
    split = protocol["split"]
    expected_split = {
        "fit_start": 0,
        "fit_end": 5644,
        "validation_start": 5644,
        "validation_end": 7056,
        "test_start": 7056,
        "test_end": 10080,
    }
    if split != expected_split:
        raise ValueError(f"split differs from the frozen protocol: {split}")
    services = protocol["services"]
    if len(services) != 10:
        raise ValueError(f"expected 10 services, found {len(services)}")
    series_names = [entry["series"] for entry in services]
    text_names = [entry["text"] for entry in services]
    if len(set(series_names)) != 10 or len(set(text_names)) != 10:
        raise ValueError("service series/text names must be unique")
    ratios = protocol["anomaly_ratios"]
    if len(ratios) != 61 or len(set(map(float, ratios))) != 61:
        raise ValueError("expected 61 unique anomaly-ratio candidates")
    bad_keys = FORBIDDEN_TUNABLE_KEYS.intersection(tunable)
    if bad_keys:
        raise ValueError(f"protocol-owned keys found in tunable config: {sorted(bad_keys)}")
    for key, value in tunable.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite tunable value: {key}")


def build_validation_pool(
    source: LocalAnomalyDetectDataSource,
    series_name: str,
    text_name: str,
    fit_end: int,
    validation_end: int,
    expected_channels: int,
) -> tuple[Dataset, int]:
    source.load_multi_list([series_name], [text_name])
    full_series = source.dataset.get_series(series_name)
    full_text = source.dataset.get_text(text_name)
    if full_series is None or full_text is None:
        raise RuntimeError(f"loader did not return the pair {series_name}, {text_name}")
    if len(full_series) != 10080 or len(full_text) != 10080:
        raise ValueError(
            f"unexpected full length for {series_name}: series={len(full_series)}, "
            f"text={len(full_text)}"
        )
    feature_count = int(full_series.shape[1] - 1)
    if feature_count != expected_channels:
        raise ValueError(
            f"channel mismatch for {series_name}: expected {expected_channels}, "
            f"found {feature_count}"
        )

    # The outer test rows [7056, 10080) are deliberately excluded here.
    series_view = full_series.iloc[:validation_end].copy(deep=True)
    text_view = full_text.iloc[:validation_end].copy(deep=True)
    metadata = source.dataset.metadata.loc[[series_name]].copy(deep=True)
    metadata.loc[series_name, "train_lens"] = fit_end
    metadata.loc[series_name, "time_steps"] = validation_end
    metadata.loc[series_name, "total_len"] = validation_end
    metadata.loc[series_name, "train/total"] = fit_end / validation_end

    pool = Dataset()
    pool.set_data(
        data_dict={series_name: series_view},
        covariate_dict={},
        metadata=metadata,
    )
    pool.update_multi_data({}, {text_name: text_view}, {})
    return pool, feature_count


def evaluate_service(
    source: LocalAnomalyDetectDataSource,
    service: dict[str, Any],
    protocol: dict[str, Any],
    tunable: dict[str, Any],
) -> list[dict[str, Any]]:
    series_name = service["series"]
    text_name = service["text"]
    split = protocol["split"]
    pool, feature_count = build_validation_pool(
        source,
        series_name,
        text_name,
        fit_end=split["fit_end"],
        validation_end=split["validation_end"],
        expected_channels=int(service["channels"]),
    )
    DataPool().set_pool(pool)

    params = dict(tunable)
    params["enc_in_time"] = feature_count
    params["anomaly_ratio"] = list(protocol["anomaly_ratios"])
    fix_random_seed(int(protocol["seed"]))

    evaluator = Evaluator([{"name": protocol["metric"]}])
    strategy = UnFixedDetectLabel(
        {"strategy_name": "unfixed_detect_label"}, evaluator
    )
    factory = ModelFactory("MindTSLogged", MindTSLogged, params)
    started = time.monotonic()
    raw_rows = strategy.multi_execute(series_name, text_name, factory)
    elapsed = time.monotonic() - started

    parsed: list[dict[str, Any]] = []
    for raw in raw_rows:
        record = dict(zip(strategy.field_names, raw))
        value = record.get(protocol["metric"])
        ratio = record.get("typical_anomaly_ratio")
        parsed.append(
            {
                "service": service["name"],
                "series": series_name,
                "text": text_name,
                "channels": feature_count,
                "anomaly_ratio": ratio,
                "affiliation_f": value,
                "elapsed_seconds": elapsed,
                "log_info": record.get("log_info", ""),
            }
        )

    finite = [
        row
        for row in parsed
        if isinstance(row["affiliation_f"], (int, float, np.floating))
        and math.isfinite(float(row["affiliation_f"]))
    ]
    if not finite:
        details = "\n".join(str(row.get("log_info", "")) for row in parsed)
        raise RuntimeError(f"no finite affiliation_f for {service['name']}\n{details}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json(args.protocol.resolve())
    tunable = load_json(args.model_config.resolve())
    validate_configuration(protocol, tunable)
    output = ensure_external_output(args.output)
    fieldnames = [
        "service",
        "series",
        "text",
        "channels",
        "anomaly_ratio",
        "affiliation_f",
        "elapsed_seconds",
        "log_info",
    ]

    source = LocalAnomalyDetectDataSource()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, service in enumerate(protocol["services"], start=1):
            print(
                json.dumps(
                    {
                        "event": "service_start",
                        "index": index,
                        "total": len(protocol["services"]),
                        "service": service["name"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            rows = evaluate_service(source, service, protocol, tunable)
            writer.writerows(rows)
            handle.flush()
            valid = [
                row
                for row in rows
                if isinstance(row["affiliation_f"], (int, float, np.floating))
                and math.isfinite(float(row["affiliation_f"]))
            ]
            best = max(valid, key=lambda row: float(row["affiliation_f"]))
            print(
                json.dumps(
                    {
                        "event": "service_complete",
                        "service": service["name"],
                        "best_f1": round(float(best["affiliation_f"]), 8),
                        "best_anomaly_ratio": float(best["anomaly_ratio"]),
                        "elapsed_seconds": round(float(best["elapsed_seconds"]), 3),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            DataPool().set_pool(None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "event": "validation_raw_complete",
                "services": len(protocol["services"]),
                "output": str(output),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
