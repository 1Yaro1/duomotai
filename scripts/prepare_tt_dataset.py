import argparse
import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "cpu_usage_system",
    "cpu_usage_total",
    "cpu_usage_user",
    "memory_usage",
    "memory_working_set",
    "rx_bytes",
    "tx_bytes",
]

FAULT_METRIC_COLUMNS = {
    "cpu_load": ["cpu_usage_system", "cpu_usage_total", "cpu_usage_user"],
    "network_delay": ["rx_bytes", "tx_bytes"],
    "network_loss": ["rx_bytes", "tx_bytes"],
}


def normalize_service_name(container_name):
    return (
        container_name.replace("dockercomposemanifests_", "")
        .replace("_1", "")
        .strip()
    )


def is_metric_csv(name):
    return (
        "/metrics/" in name
        and name.endswith(".csv")
        and "__MACOSX/" not in name
        and "/._" not in name
    )


def read_metric_csv(file_obj, service_name):
    df = pd.read_csv(file_obj)
    df = df[["timestamp"] + METRIC_COLUMNS]
    df = df.groupby("timestamp", as_index=False).mean()
    df = df.set_index("timestamp")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.rename(
        columns={metric: f"{service_name}.{metric}" for metric in METRIC_COLUMNS}
    )


def finalize_session_frame(frames, all_columns=None):
    if not frames:
        raise ValueError("No metric CSV files were found for one TT session.")

    frame = pd.concat(frames, axis=1).sort_index()
    if all_columns is not None:
        frame = frame.reindex(columns=all_columns)
    frame = frame.interpolate(limit_direction="both").ffill().bfill().fillna(0)
    return frame


def load_zip_metric_session(zip_file, session_metrics_prefix, all_columns=None):
    frames = []
    for entry in zip_file.infolist():
        if entry.filename.startswith(session_metrics_prefix) and is_metric_csv(
            entry.filename
        ):
            service_name = Path(entry.filename).stem
            frames.append(read_metric_csv(zip_file.open(entry), service_name))
    return finalize_session_frame(frames, all_columns)


def load_tar_metric_session(zip_file, tar_entry_name, all_columns=None):
    tar_bytes = zip_file.read(tar_entry_name)
    frames = []

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:xz") as tar_file:
        for member in tar_file.getmembers():
            if member.isfile() and is_metric_csv(member.name):
                service_name = Path(member.name).stem
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    continue
                frames.append(read_metric_csv(extracted, service_name))

    return finalize_session_frame(frames, all_columns)


def robust_stats(normal_frame):
    stats = {}
    for column in normal_frame.columns:
        series = (
            normal_frame[column]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        median = series.median()
        mad = (series - median).abs().median()
        stats[column] = {
            "median": median,
            "mad": mad if mad > 1e-9 else np.nan,
            "q001": series.quantile(0.001),
            "q999": series.quantile(0.999),
        }
    return stats


def feature_anomaly_mask(series, stats, z_threshold):
    values = series.astype(float)

    z_mask = pd.Series(False, index=values.index)
    mad = stats["mad"]
    if not pd.isna(mad) and mad > 1e-9:
        robust_z = (values - stats["median"]).abs() / (1.4826 * mad)
        z_mask = robust_z > z_threshold

    quantile_mask = (values < stats["q001"]) | (values > stats["q999"])
    return z_mask | quantile_mask


def smooth_boolean_mask(mask, min_run_seconds, max_gap_seconds):
    values = np.asarray(mask, dtype=bool).copy()

    idx = 0
    while idx < len(values):
        if not values[idx]:
            idx += 1
            continue
        end = idx
        while end < len(values) and values[end]:
            end += 1
        if end - idx < min_run_seconds:
            values[idx:end] = False
        idx = end

    idx = 0
    while idx < len(values):
        if values[idx]:
            idx += 1
            continue
        end = idx
        while end < len(values) and not values[end]:
            end += 1
        if idx > 0 and end < len(values) and end - idx <= max_gap_seconds:
            values[idx:end] = True
        idx = end

    return values


def fault_metric_columns(fault, frame_columns, metric_scope):
    service_name = normalize_service_name(fault["name"])

    if metric_scope == "fault_type":
        metric_names = FAULT_METRIC_COLUMNS.get(fault["fault"], METRIC_COLUMNS)
    else:
        metric_names = METRIC_COLUMNS

    columns = [
        f"{service_name}.{metric_name}"
        for metric_name in metric_names
        if f"{service_name}.{metric_name}" in frame_columns
    ]
    return service_name, columns


def label_fault_session(
    frame,
    faults,
    stats,
    metric_scope,
    z_threshold,
    min_run_seconds,
    max_gap_seconds,
    post_fault_tail_seconds,
):
    label = pd.Series(False, index=frame.index)
    details = []

    for fault in faults:
        service_name, columns = fault_metric_columns(fault, frame.columns, metric_scope)
        start = int(fault["start"])
        end = int(fault["start"] + fault["duration"] + post_fault_tail_seconds)
        in_candidate_window = (frame.index >= start) & (frame.index <= end)

        if not columns or not in_candidate_window.any():
            details.append(
                {
                    "service": service_name,
                    "fault": fault["fault"],
                    "injection_start": start,
                    "injection_end": int(fault["start"] + fault["duration"]),
                    "label_start": "",
                    "label_end": "",
                    "raw_anomaly_seconds": 0,
                    "smoothed_anomaly_seconds": 0,
                }
            )
            continue

        candidate_frame = frame.loc[in_candidate_window, columns]
        masks = [
            feature_anomaly_mask(candidate_frame[column], stats[column], z_threshold)
            for column in columns
        ]
        raw_mask = pd.concat(masks, axis=1).any(axis=1)
        smoothed = pd.Series(
            smooth_boolean_mask(raw_mask, min_run_seconds, max_gap_seconds),
            index=raw_mask.index,
        )
        label.loc[smoothed.index] |= smoothed

        labeled_index = smoothed.index[smoothed]
        details.append(
            {
                "service": service_name,
                "fault": fault["fault"],
                "injection_start": start,
                "injection_end": int(fault["start"] + fault["duration"]),
                "label_start": int(labeled_index.min()) if len(labeled_index) else "",
                "label_end": int(labeled_index.max()) if len(labeled_index) else "",
                "raw_anomaly_seconds": int(raw_mask.sum()),
                "smoothed_anomaly_seconds": int(smoothed.sum()),
            }
        )

    return label.astype(int), details


def resample_session(frame, labels, seconds):
    if seconds <= 1:
        return frame.reset_index(drop=True), labels.reset_index(drop=True)

    bucket = np.arange(len(frame)) // seconds
    resampled_frame = frame.groupby(bucket).mean()
    resampled_labels = labels.groupby(bucket).max()
    resampled_frame.index = pd.RangeIndex(len(resampled_frame))
    resampled_labels.index = pd.RangeIndex(len(resampled_labels))
    return resampled_frame, resampled_labels.astype(int)


def write_long_metric_csv(path, frame, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = range(1, len(frame) + 1)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["date", "data", "cols"])

        for column in frame.columns:
            values = frame[column].to_numpy()
            for date, value in zip(dates, values):
                writer.writerow([date, f"{float(value):.10g}", column])

        for date, value in zip(dates, labels.to_numpy()):
            writer.writerow([date, int(value), "label"])


def write_blank_text_csv(path, length, placeholder):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["date", "data", "cols"])
        for date in range(1, length + 1):
            writer.writerow([date, placeholder, "channel1"])


def update_metadata(metadata_path, file_name, train_length, total_length):
    row = {
        "file_name": file_name,
        "trend": "FALSE",
        "seasonal": "FALSE",
        "stationary": "FALSE",
        "pattern": "FALSE",
        "shifting": "TRUE",
        "dataset_name": "TT",
        "train_lens": int(train_length),
        "time_steps": int(total_length),
        "if_univariate": "FALSE",
        "size": "large",
        "type_value": "microservice_metrics",
        "total_len": int(total_length),
        "train/total": f"{float(train_length) / float(total_length):.6f}",
    }

    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    columns = next(csv.reader([lines[0]]))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="")
    writer.writerow([row.get(column, "") for column in columns])
    new_line = output.getvalue()

    kept_lines = [lines[0]]
    for line in lines[1:]:
        parsed = next(csv.reader([line]))
        if parsed and parsed[0] == file_name:
            continue
        kept_lines.append(line)

    kept_lines.append(new_line)
    metadata_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the TT metric dataset for MindTS anomaly detection."
    )
    parser.add_argument(
        "--zip-path",
        default=r"D:\dataset\TT Dataset.zip",
        help="Path to TT Dataset.zip.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/anomaly_detect",
        help="MindTS anomaly detection dataset directory.",
    )
    parser.add_argument("--data-name", default="TT.csv")
    parser.add_argument("--text-name", default="TT_text.csv")
    parser.add_argument(
        "--resample-seconds",
        type=int,
        default=10,
        help="Aggregate raw 1-second metrics into this many seconds per model step.",
    )
    parser.add_argument(
        "--metric-scope",
        choices=["service", "fault_type"],
        default="service",
        help="Use all affected-service metrics, or only metrics matching the fault type.",
    )
    parser.add_argument("--z-threshold", type=float, default=8.0)
    parser.add_argument("--min-run-seconds", type=int, default=5)
    parser.add_argument("--max-gap-seconds", type=int, default=10)
    parser.add_argument(
        "--post-fault-tail-seconds",
        type=int,
        default=120,
        help="Allow delayed metric anomalies after the injected fault ends.",
    )
    parser.add_argument(
        "--text-placeholder",
        default=".",
        help="Placeholder token for the intentionally empty text dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_path = output_dir / "data" / args.data_name
    text_path = output_dir / "data" / args.text_name
    metadata_path = output_dir / "DETECT_META.csv"
    summary_path = output_dir / "TT_LABEL_SUMMARY.csv"

    with zipfile.ZipFile(args.zip_path) as zip_file:
        normal_entries = sorted(
            entry.filename
            for entry in zip_file.infolist()
            if entry.filename.startswith("TT Dataset/no fault/TT.")
            and entry.filename.endswith(".tar.xz")
            and "/._" not in entry.filename
        )
        if not normal_entries:
            raise ValueError("No no-fault TT metric archives were found.")

        normal_frames = [
            load_tar_metric_session(zip_file, entry) for entry in normal_entries
        ]
        all_columns = sorted(set().union(*(set(frame.columns) for frame in normal_frames)))
        normal_frames = [
            frame.reindex(columns=all_columns)
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .fillna(0)
            for frame in normal_frames
        ]
        normal_for_stats = pd.concat(normal_frames, axis=0)
        stats = robust_stats(normal_for_stats)

        resampled_normal_frames = []
        resampled_normal_labels = []
        for frame in normal_frames:
            labels = pd.Series(0, index=frame.index)
            resampled_frame, resampled_label = resample_session(
                frame, labels, args.resample_seconds
            )
            resampled_normal_frames.append(resampled_frame)
            resampled_normal_labels.append(resampled_label)

        fault_json_entries = sorted(
            entry.filename
            for entry in zip_file.infolist()
            if entry.filename.startswith("TT Dataset/data/TT.fault-")
            and entry.filename.endswith(".json")
            and "/._" not in entry.filename
        )
        if not fault_json_entries:
            raise ValueError("No TT fault JSON files were found.")

        resampled_fault_frames = []
        resampled_fault_labels = []
        summary_rows = []

        for fault_json_entry in fault_json_entries:
            session_name = (
                Path(fault_json_entry)
                .name.replace("TT.fault-", "TT.")
                .replace(".json", "")
            )
            metrics_prefix = f"TT Dataset/data/{session_name}/metrics/"
            frame = load_zip_metric_session(zip_file, metrics_prefix, all_columns)
            faults = json.loads(zip_file.read(fault_json_entry))["faults"]
            labels, details = label_fault_session(
                frame=frame,
                faults=faults,
                stats=stats,
                metric_scope=args.metric_scope,
                z_threshold=args.z_threshold,
                min_run_seconds=args.min_run_seconds,
                max_gap_seconds=args.max_gap_seconds,
                post_fault_tail_seconds=args.post_fault_tail_seconds,
            )
            for detail in details:
                detail["session"] = session_name
                summary_rows.append(detail)

            resampled_frame, resampled_label = resample_session(
                frame, labels, args.resample_seconds
            )
            resampled_fault_frames.append(resampled_frame)
            resampled_fault_labels.append(resampled_label)

    train_frame = pd.concat(resampled_normal_frames, ignore_index=True)
    train_label = pd.concat(resampled_normal_labels, ignore_index=True)
    test_frame = pd.concat(resampled_fault_frames, ignore_index=True)
    test_label = pd.concat(resampled_fault_labels, ignore_index=True)

    combined_frame = pd.concat([train_frame, test_frame], ignore_index=True)
    combined_label = pd.concat([train_label, test_label], ignore_index=True)
    combined_frame = combined_frame[all_columns].fillna(0)

    write_long_metric_csv(data_path, combined_frame, combined_label)
    write_blank_text_csv(text_path, len(combined_frame), args.text_placeholder)
    update_metadata(metadata_path, args.data_name, len(train_frame), len(combined_frame))
    write_summary(summary_path, summary_rows)

    print(f"Wrote {data_path}")
    print(f"Wrote {text_path}")
    print(f"Updated {metadata_path}")
    print(f"Wrote {summary_path}")
    print(f"Train steps: {len(train_frame)}")
    print(f"Total steps: {len(combined_frame)}")
    print(f"Time-series channels: {len(all_columns)}")
    print(f"Anomaly steps after resampling: {int(combined_label.sum())}")


if __name__ == "__main__":
    main()
