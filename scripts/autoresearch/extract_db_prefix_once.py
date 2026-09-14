"""One authorized scan per original CSV. Subsequent readers use only NPZ.

Suffix bytes enter bounded I/O buffers but are never decoded as CSV fields,
numeric values, labels, or retained in any output/digest/statistic.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def select_prefix_stream(stream, end=7056):
    header = stream.readline()
    if header.rstrip(b"\r\n") != b"date,data,cols":
        raise ValueError("Unexpected structural header")
    selected = bytearray(header)
    ordinal = bytearray()
    in_body, retain = False, False
    # No splitting full suffix lines into fields. Only ordinal is decoded.
    while True:
        block = stream.read(65536)
        if not block:
            break
        for byte in block:
            if not in_body:
                if byte == 44:
                    if not ordinal or not ordinal.isdigit():
                        raise ValueError("Invalid structural row ID")
                    row_id = int(ordinal)
                    if not 1 <= row_id <= 10080:
                        raise ValueError("Unexpected structural row ID")
                    retain = row_id <= end
                    if retain:
                        selected.extend(ordinal)
                        selected.append(byte)
                    ordinal.clear()
                    in_body = True
                else:
                    if len(ordinal) >= 5 or not 48 <= byte <= 57:
                        raise ValueError("Unexpected row-ID encoding; refusing full-line parsing")
                    ordinal.append(byte)
            else:
                if retain:
                    selected.append(byte)
                if byte == 10:
                    in_body = False
    if ordinal:
        raise ValueError("Incomplete structural row ID")
    return bytes(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.run / "run_config.json").read_text())["identity"]
    from ts_benchmark.data.utils import process_data_df
    manifest = dict(suffix_bytes_scanned=True, test_values_parsed=False,
                    test_values_materialized=False, test_values_saved=False,
                    test_values_used=False, test_labels_used=False, services={})
    for name in ("GAIA_dbservice1", "GAIA_dbservice2"):
        expected = identity["input_prefixes"][name]
        path = args.raw_dir / (name + ".csv")
        # Exclusive receipt precedes the only source open. Do not auto retry.
        with (args.output / (name + ".source_open_receipt.json")).open("x") as receipt:
            json.dump(dict(source=str(path), open_budget=1, automatic_retry_allowed=False), receipt)
        with path.open("rb") as stream:
            prefix_csv = select_prefix_stream(stream)
        raw = pd.read_csv(io.BytesIO(prefix_csv))
        names = raw["cols"].unique().tolist()
        assert names == expected["columns"] and names[-1] == "label"
        assert len(raw) == 7056 * len(names)
        for i, col in enumerate(names):
            block = raw.iloc[i*7056:(i+1)*7056]
            assert (block["cols"] == col).all()
            assert np.array_equal(block["date"].to_numpy(), np.arange(1, 7057))
        frame = process_data_df(raw)
        values = np.ascontiguousarray(frame.to_numpy())
        assert values.shape == (7056, len(names)) and np.isfinite(values).all()
        assert list(frame.columns) == expected["columns"]
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        assert digest == expected["sha256"], "Historical prefix byte digest mismatch"
        with np.load(args.run / (name + ".validation_arrays.npz"), allow_pickle=False) as history:
            assert np.array_equal(values[5644:7056, -1], history["actual"])
        cache_path = args.output / (name + ".prefix.npz")
        with cache_path.open("xb") as cache:
            np.savez_compressed(cache, values=values, columns=np.array(names),
                                absolute_index=np.arange(7056))
        # Round trip is a literal byte/element comparison against extracted prefix.
        with np.load(cache_path, allow_pickle=False) as saved:
            assert np.array_equal(saved["values"], values)
            assert saved["values"].tobytes() == values.tobytes()
        manifest["services"][name] = dict(rows=7056, metric_channels=len(names)-1,
            columns=names, prefix_sha256=digest, historical_sha256_match=True,
            cache_roundtrip_elementwise_equal=True, historical_validation_labels_equal=True,
            historical_full_prefix_direct_elementwise_comparison=None,
            identity_evidence="historical contiguous-array SHA256; historical full array not archived",
            cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest())
        cache_path.chmod(0o444)
    manifest_path = args.output / "prefix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o444)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
