"""Versioned GAIA business-log sidecars. Never overwrite the legacy CSVs.

extract: stream the source ZIP to a lossless, train/validation-only SQLite store.
build: fit Drain3 on [0,4515), freeze, and cache counts plus frozen BERT vectors.
This is preprocessing, not anomaly-model training or evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
import csv
from datetime import datetime
from functools import lru_cache
import hashlib
import io
from importlib.metadata import version
import json
from pathlib import Path
import re
import sqlite3
import time
import zipfile

SERVICES = tuple(f"{kind}service{n}" for kind in ("db", "web", "log", "mob", "redis") for n in (1, 2))
START = datetime(2021, 8, 24)
FIT_END, POOL_END, END = 4515, 5644, 7056
MEMBER = "business/business_table_2021-08.csv"
# Same explicit control-message exclusions as the historical business pipeline.
# Keep excluded messages in the raw sidecar; never infer exclusions from labels.
CONTROL = (
    "[memory_anomalies]", "[cpu_anomalies]", "trigger a high memory program",
    "trigger a parallel fast sorting program", "trigger the file moving program",
    "trigger an access permission denied exception", "simulate the login failure",
    "normal memory freed label",
)
MASKS = (
    (r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b", "<UUID>"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>"),
    (r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{16,})\b", "<HEX>"),
    (r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "<JWT>"),
    (r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)", "<NUM>"),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


@contextmanager
def raw_connection(path):
    # SQLite URI read-only: a typo cannot create an empty database.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        yield db


def event_position(message):
    # Leading formatting whitespace is not part of the timestamp. Keep the
    # original message unchanged in the raw store.
    match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:[,.](\d+))?", message.lstrip())
    if not match:
        raise ValueError("Business message has no valid leading timestamp")
    stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
    delta = (stamp - START).total_seconds()
    fraction = float("0." + match[2]) if match[2] else 0.0
    return int(delta // 60), (stamp.second + fraction) / 60.0


@lru_cache(maxsize=65536)
def normalize(message):
    # Preserve severity and full message body; no character/token truncation.
    # Complete unmodified message remains available in events.sqlite.
    parts = message.split("|", 6)
    body = (parts[1].strip() + " " + parts[-1]) if len(parts) == 7 else message
    for pattern, replacement in MASKS:
        body = re.sub(pattern, replacement, body)
    return re.sub(r"\s+", " ", body).strip() or "<EMPTY>"


def extract(source, output, progress_every=1_000_000):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source)
    db_path = output / "events.sqlite"
    quality, started = Counter(), time.monotonic()
    with closing(sqlite3.connect(db_path)) as db, zipfile.ZipFile(source) as archive:
        db.execute("CREATE TABLE events (source_order INTEGER PRIMARY KEY, record_id TEXT, "
                   "service TEXT, minute INTEGER CHECK(minute>=0 AND minute<7056), "
                   "position REAL, message TEXT, excluded INTEGER CHECK(excluded IN (0,1)))")
        db.execute("CREATE TABLE empty_messages (source_order INTEGER PRIMARY KEY, record_id TEXT, "
                   "service TEXT, source_date TEXT, message TEXT, reason TEXT)")
        # csv.reader handles quoted commas and multi-line messages correctly.
        csv.field_size_limit(32 * 1024 * 1024)
        with archive.open(MEMBER) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""), strict=True)
            if reader.fieldnames != ["id", "datetime", "service", "message"]:
                raise ValueError("Unexpected source CSV columns")
            batch = []
            for ordinal, row in enumerate(reader):
                if None in row or any(row[k] is None for k in reader.fieldnames):
                    raise ValueError(f"Malformed source record {ordinal}")
                # Source ZIP contains other dates. Gate by date/timestamp before
                # inspecting business content; never materialize test features.
                if row["service"] in SERVICES and "2021-08-24" <= row["datetime"][:10] <= "2021-08-28":
                    if not row["message"].strip():
                        # An empty database row has no log event or minute to
                        # recover. Preserve it separately, never invent a time.
                        db.execute("INSERT INTO empty_messages VALUES (?,?,?,?,?,?)",
                                   (ordinal, row["id"], row["service"], row["datetime"],
                                    row["message"], "empty_message_no_event_timestamp"))
                        quality["empty_message_rows_preserved"] += 1
                        continue
                    try:
                        minute, position = event_position(row["message"])
                    except ValueError:
                        write_json(output / "unresolved_record.json", {"source_order": ordinal, **row})
                        raise ValueError(f"Nonempty message lacks timestamp at source record {ordinal}; "
                                         "saved unresolved_record.json; cache remains unusable") from None
                    if 0 <= minute < END:
                        excluded = int(any(x in row["message"].lower() for x in CONTROL))
                        batch.append((ordinal, row["id"], row["service"], minute, position,
                                      row["message"], excluded))
                        quality[row["service"]] += 1
                        quality["control_messages_preserved"] += excluded
                if len(batch) >= 10000:
                    db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch)
                    db.commit()
                    batch.clear()
                if (ordinal + 1) % progress_every == 0:
                    print(json.dumps({"stage": "extract", "scanned_records": ordinal + 1,
                                      "last_source_date": row["datetime"][:10],
                                      "uncompressed_bytes_read": binary.tell(),
                                      "kept_by_service": dict(quality),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            if batch:
                db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", batch)
        db.execute("CREATE INDEX event_time ON events(service,minute,position,source_order)")
        db.commit()
    if any(quality[s] == 0 for s in SERVICES):
        raise ValueError("Missing service: extraction incomplete, no complete manifest written")
    # Detect source changes during extraction, not just before it.
    if sha256(source) != source_hash:
        raise RuntimeError("Source archive changed during extraction")
    result = {"schema_version": 2, "complete": True, "source_sha256": source_hash,
              "extractor_sha256": sha256(__file__),
              "source_member": MEMBER, "timezone": "Asia/Shanghai", "start": START.isoformat(),
              "end_exclusive": END, "services": list(SERVICES), "counts": dict(quality),
              "database_sha256": sha256(db_path), "test_features_created": False}
    write_json(output / "source.json", result)
    print(json.dumps({"stage": "extract", "complete": True, "output": str(output)}), flush=True)


def make_miner():
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.6
    cfg.drain_depth = 4
    cfg.drain_max_clusters = None  # No eviction of rare or old templates.
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def template_snapshot(miner):
    return [(int(c.cluster_id), c.get_template(), int(c.size)) for c in miner.drain.clusters]


def build_service(db, service, embed):
    import numpy as np
    if service not in SERVICES:
        raise ValueError("Unknown service")
    miner = make_miner()
    print(json.dumps({"stage": "fit_templates", "service": service, "fit_end_exclusive": FIT_END}), flush=True)
    query = "SELECT minute,message FROM events WHERE service=? AND excluded=0 AND minute<? ORDER BY minute,position,source_order"
    fit_events = 0
    for minute, message in db.execute(query, (service, FIT_END)):
        miner.add_log_message(normalize(message))
        fit_events += 1
        if fit_events % 100000 == 0:
            print(json.dumps({"stage": "fit_templates", "service": service, "fit_events": fit_events}), flush=True)
    if fit_events == 0:
        raise ValueError(f"No gradient-training business events for {service}")
    before = template_snapshot(miner)
    id_map = {cid: i + 1 for i, (cid, _, _) in enumerate(before)}
    vocabulary = ["[UNK]"] + [template for _, template, _ in before]
    counts = np.zeros((END, len(vocabulary)), dtype=np.int64)

    @lru_cache(maxsize=65536)
    def match(text):
        found = miner.match(text, full_search_strategy="always")
        return 0 if found is None else id_map[found.cluster_id]

    for minute, message in db.execute(query, (service, END)):
        if not 0 <= minute < END:
            raise ValueError("Out-of-scope minute")
        counts[minute, match(normalize(message))] += 1
    if before != template_snapshot(miner):
        raise RuntimeError("Inference mutated the Drain vocabulary")
    semantics = np.asarray(embed(vocabulary), dtype=np.float32)
    if semantics.ndim != 2 or semantics.shape[0] != len(vocabulary) or not np.isfinite(semantics).all():
        raise ValueError("Invalid semantic embeddings")
    log_count = np.log1p(counts[:FIT_END]).astype(np.float32)
    mean = log_count.mean(axis=0)
    std = log_count.std(axis=0)
    std[std < 1e-6] = 1.0
    present = counts.sum(axis=1) > 0
    arrays = dict(counts=counts, semantics=semantics, count_mean=mean, count_std=std,
                  minute=np.arange(END, dtype=np.int64), present=present)
    quality = {"fit_events": fit_events, "templates": len(before),
               "fit_end_exclusive": FIT_END, "rows": END,
               "events": int(counts.sum()), "missing_minutes": int((~present).sum()),
               "unknown_events": int(counts[:, 0].sum()),
               "unknown_validation_events": int(counts[POOL_END:, 0].sum()),
               "vocabulary_frozen": True}
    return arrays, vocabulary, quality


class FrozenBert:
    """Frozen, local-only BERT; long templates are chunked, never truncated."""
    def __init__(self, model_path, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.path = Path(model_path).resolve(strict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=False).to(device)
        if self.model.config.model_type != "bert":
            raise ValueError("Log v2 requires BERT; do not silently substitute another model")
        self.model.eval().requires_grad_(False)

    def __call__(self, texts):
        import numpy as np
        vectors = []
        capacity = self.model.config.max_position_embeddings - 2
        with self.torch.inference_mode():
            for text in texts:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens or [self.tokenizer.unk_token_id]
                accum, total = None, 0
                for start in range(0, len(tokens), capacity):
                    part = tokens[start:start + capacity]
                    ids = self.tokenizer.build_inputs_with_special_tokens(part)
                    ids = self.torch.tensor([ids], device=self.device)
                    hidden = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids)).last_hidden_state
                    value = hidden[0, 1:1 + len(part)].float().sum(0)
                    accum = value if accum is None else accum + value
                    total += len(part)
                vectors.append((accum / total).cpu().numpy())
        return np.stack(vectors).astype(np.float32)

    def provenance(self):
        # Pin actual local weights/tokenizer/config, not just a mutable model name.
        files = [p for p in self.path.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".bin", ".safetensors")]
        return {"model_path": str(self.path), "files_sha256": {p.name: sha256(p) for p in files},
                "type": "bert", "frozen": True, "pooling": "all_non_special_tokens_chunked_mean",
                "hidden_size": self.model.config.hidden_size, "svd": False}


def build(raw_dir, output, encoder):
    import numpy as np
    raw_dir, output = Path(raw_dir), Path(output)
    source = json.loads((raw_dir / "source.json").read_text(encoding="utf-8"))
    if not source.get("complete") or source["end_exclusive"] != END or source["services"] != list(SERVICES):
        raise ValueError("Raw extraction is incomplete or uses another scope")
    if sha256(raw_dir / "events.sqlite") != source["database_sha256"]:
        raise ValueError("Raw sidecar changed")
    output.mkdir(parents=True, exist_ok=False)
    services = {}
    with raw_connection(raw_dir / "events.sqlite") as db:
        for service in SERVICES:
            arrays, vocabulary, quality = build_service(db, service, encoder)
            filename = f"GAIA_{service}.npz"
            np.savez_compressed(output / filename, **arrays)
            write_json(output / f"GAIA_{service}.vocabulary.json", vocabulary)
            services["GAIA_" + service] = dict(quality, file=filename, sha256=sha256(output / filename),
                vocabulary_sha256=sha256(output / f"GAIA_{service}.vocabulary.json"))
            print(json.dumps({"stage": "build", "service": service, **quality}), flush=True)
    manifest = {"schema_version": 2, "complete": True, "seed": 2021,
                "fit_end_exclusive": FIT_END, "pool_end_exclusive": POOL_END,
                "end_exclusive": END, "start": START.isoformat(), "timezone": "Asia/Shanghai",
                "source": source, "semantics": encoder.provenance(), "services": services,
                "test_features_created": False,
                "preparer_sha256": sha256(__file__), "drain3_version": version("drain3"),
                "control_patterns": list(CONTROL), "mask_rules": list(MASKS),
                "representation": "all_template_counts_and_count_weighted_minute_semantics",
                "normalization": "log1p_counts_standardized_on_0_4515_only"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"stage": "build", "complete": True, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--source", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw-dir", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--bert-path", type=Path, required=True)
    b.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.source, args.output)
    else:
        build(args.raw_dir, args.output, FrozenBert(args.bert_path, args.device))


if __name__ == "__main__":
    main()
