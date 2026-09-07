"""Opt-in minute-aligned log v2 inputs. No labels, token truncation or test rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

START = pd.Timestamp("2021-08-24")
FIT_END, POOL_END, END = 4515, 5644, 7056
SERVICES = tuple(f"GAIA_{kind}service{n}" for kind in ("db", "web", "log", "mob", "redis") for n in (1, 2))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class LogV2Store:
    def __init__(self, cache_dir, service):
        root = Path(cache_dir).resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("schema_version") != 2:
            raise ValueError("Incomplete log v2 cache")
        if (manifest.get("fit_end_exclusive"), manifest.get("pool_end_exclusive"),
            manifest.get("end_exclusive"), manifest.get("seed")) != (FIT_END, POOL_END, END, 2021):
            raise ValueError("Log cache split/seed mismatch")
        if manifest.get("test_features_created") is not False or set(manifest["services"]) != set(SERVICES):
            raise ValueError("Log cache test/service scope mismatch")
        if manifest.get("start") != START.isoformat() or manifest.get("timezone") != "Asia/Shanghai":
            raise ValueError("Log cache time origin mismatch")
        if service not in SERVICES:
            raise ValueError("Unknown service")
        record = manifest["services"][service]
        path = (root / record["file"]).resolve(strict=True)
        if path.parent != root or digest(path) != record["sha256"]:
            raise ValueError("Invalid or changed service cache")
        vocab_path = root / f"{service}.vocabulary.json"
        if digest(vocab_path) != record["vocabulary_sha256"]:
            raise ValueError("Vocabulary changed")
        with np.load(path, allow_pickle=False) as data:
            self.counts = data["counts"].copy()
            self.semantics = data["semantics"].copy()
            self.mean, self.std = data["count_mean"].copy(), data["count_std"].copy()
            self.present, minute = data["present"].copy(), data["minute"].copy()
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        if self.counts.ndim != 2 or self.counts.shape[0] != END or not np.array_equal(minute, np.arange(END)):
            raise ValueError("Rows reordered, missing, or extended into test")
        g = self.counts.shape[1]
        if self.counts.dtype.kind not in "iu" or (self.counts < 0).any():
            raise ValueError("Counts must be nonnegative integers")
        if (self.semantics.ndim != 2 or self.semantics.shape[0] != g or len(vocabulary) != g
                or self.mean.shape != (g,) or self.std.shape != (g,)
                or self.present.shape != (END,)):
            raise ValueError("Inconsistent feature dimensions")
        if not all(np.isfinite(x).all() for x in (self.semantics, self.mean, self.std)) or (self.std <= 0).any():
            raise ValueError("Non-finite features or invalid scale")
        if not np.array_equal(self.present, self.counts.sum(1) > 0):
            raise ValueError("Presence mask does not match event counts")
        # Independently recompute train-only moments to catch full-period fitting.
        train = np.log1p(self.counts[:FIT_END]).astype(np.float32)
        expected_std = train.std(0)
        expected_std[expected_std < 1e-6] = 1.0
        if not np.allclose(self.mean, train.mean(0), atol=1e-6) or not np.allclose(self.std, expected_std, atol=1e-6):
            raise ValueError("Count scaler was not fitted on gradient training only")
        self.semantic_dim, self.count_dim = self.semantics.shape[1], g

    def window(self, start, length):
        if not isinstance(start, (int, np.integer)) or length <= 0 or start < 0 or start + length > END:
            raise ValueError("Window outside approved train/validation interval")
        counts = self.counts[start:start + length].astype(np.float32)
        total = counts.sum(1, keepdims=True)
        semantic = (counts @ self.semantics) / np.maximum(total, 1.0)
        scaled = (np.log1p(counts) - self.mean) / self.std
        features = np.concatenate([semantic, scaled], axis=1).astype(np.float32)
        return features, self.present[start:start + length].astype(np.float32)


class LogV2Dataset(Dataset):
    def __init__(self, data, store, win_size=24, step=1, mode="train"):
        if mode not in {"train", "val", "test", "thre"} or win_size != 24 or step != 1:
            raise ValueError("Keep the frozen window=24, step=1 and loader modes")
        index = pd.DatetimeIndex(data.index)
        if index.tz is not None:
            index = index.tz_convert("Asia/Shanghai").tz_localize(None)
        minute_float = np.asarray((index - START).total_seconds()) / 60
        if len(index) < win_size or not np.isfinite(minute_float).all():
            raise ValueError("Invalid numeric time index")
        ids = minute_float.astype(np.int64)
        if not np.array_equal(minute_float, ids) or not np.all(np.diff(ids) == 1):
            raise ValueError("Numeric minutes must be consecutive and ordered")
        if ids[0] < 0 or ids[-1] >= END:
            raise ValueError("Refusing test or out-of-period data")
        # A loader must not cross either held-out boundary.
        if any(ids[0] < end <= ids[-1] for end in (FIT_END, POOL_END)):
            raise ValueError("Window pool crosses a fixed split boundary")
        self.data = np.asarray(data.values, dtype=np.float32)
        if not np.isfinite(self.data).all():
            raise ValueError("Non-finite numeric inputs")
        self.first, self.store, self.win = int(ids[0]), store, win_size
        self.step = win_size if mode == "thre" else step

    def __len__(self):
        return (len(self.data) - self.win) // self.step + 1

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError(i)
        offset = i * self.step
        features, present = self.store.window(self.first + offset, self.win)
        # Fourth field is unused by reconstruction training; no true labels read.
        return (self.data[offset:offset + self.win].copy(), features, present,
                np.zeros(self.win, dtype=np.float32))


def log_v2_provider(data, text, batch_size, win_size=24, step=1, mode="train", *, store):
    dataset = LogV2Dataset(data, store, win_size, step, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=mode in {"train", "val"},
                      num_workers=0, drop_last=False)


class MinuteLogEncoder(nn.Module):
    """Keep 24 minute tokens; missingness is a learned signal, not row removal."""
    def __init__(self, semantic_dim, count_dim, d_model, seq_len=24):
        super().__init__()
        if min(semantic_dim, count_dim, d_model) < 1 or seq_len != 24:
            raise ValueError("Invalid log encoder dimensions")
        self.semantic_dim, self.count_dim, self.seq_len = semantic_dim, count_dim, seq_len
        self.semantic = nn.Linear(semantic_dim, d_model)
        self.count = nn.Linear(count_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.missing = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        # Distinct initial positions, deterministic under the fixed torch seed.
        nn.init.normal_(self.position, std=0.02)

    def forward(self, features, present):
        if (features.ndim != 3 or features.shape[1:] != (self.seq_len, self.semantic_dim + self.count_dim)
                or present.shape != features.shape[:2]):
            raise ValueError("Expected log features [B,24,D+G] and presence [B,24]")
        if not torch.isfinite(features).all() or not torch.isfinite(present).all() or not torch.all((present == 0) | (present == 1)):
            raise ValueError("Invalid log features/presence")
        semantic = self.semantic(features[..., :self.semantic_dim]) * present.unsqueeze(-1)
        count = self.count(features[..., self.semantic_dim:])
        return self.norm(semantic + count + self.position + (1 - present.unsqueeze(-1)) * self.missing)
