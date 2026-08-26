#!/usr/bin/env python3
"""Read raw GAIA validation records and emit the frozen equal-weight metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise TypeError("protocol must be a JSON object")
    if value.get("metric") != "affiliation_f":
        raise ValueError("protocol metric must remain affiliation_f")
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"service", "anomaly_ratio", "affiliation_f", "log_info"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"raw result is missing columns: {sorted(missing)}")
        return list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol.resolve())
    rows = load_rows(args.input.resolve())
    expected_services = [entry["name"] for entry in protocol["services"]]
    expected_ratios = [float(value) for value in protocol["anomaly_ratios"]]
    expected_ratio_set = set(expected_ratios)
    seen_services = {row["service"] for row in rows}
    if seen_services != set(expected_services):
        raise ValueError(
            f"service mismatch: missing={sorted(set(expected_services) - seen_services)}, "
            f"extra={sorted(seen_services - set(expected_services))}"
        )

    best_values: list[float] = []
    for service in expected_services:
        service_rows = [row for row in rows if row["service"] == service]
        ratios = [float(row["anomaly_ratio"]) for row in service_rows]
        if len(service_rows) != len(expected_ratios) or set(ratios) != expected_ratio_set:
            raise ValueError(
                f"{service} candidate mismatch: expected {len(expected_ratios)} unique "
                f"ratios, found {len(service_rows)} rows and {len(set(ratios))} unique ratios"
            )
        valid: list[tuple[int, float, float]] = []
        for order, row in enumerate(service_rows):
            try:
                value = float(row["affiliation_f"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                valid.append((order, value, float(row["anomaly_ratio"])))
        if not valid:
            logs = "\n".join(row.get("log_info", "") for row in service_rows)
            raise ValueError(f"{service} has no finite affiliation_f\n{logs}")
        # Match pandas idxmax: retain the first candidate when several values tie.
        best = max(valid, key=lambda item: item[1])
        best_values.append(best[1])
        print(
            f'{{"service":"{service}","best_f1":{best[1]:.8f},'
            f'"best_anomaly_ratio":{best[2]:.8g}}}',
            flush=True,
        )

    mean_best_f1 = math.fsum(best_values) / len(expected_services)
    print(f'{{"mean_best_f1": {mean_best_f1:.4f}}}', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
