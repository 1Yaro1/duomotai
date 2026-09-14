"""Opt-in, checkpoint-compatible score protocol; never trains or opens raw CSV.

Full-window normalization and statistical prompts remain historical references.
Masked-only scoring does NOT remove their access to masked input statistics.
"""
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import Dataset

MASK_BANK = tuple(tuple(i in pair for i in range(4)) for pair in combinations(range(4), 2))
PROTOCOL = "deterministic_overlap_v1"


def apply_fixed_patch_mask(tokens, mask):
    mask = torch.as_tensor(mask, device=tokens.device, dtype=torch.bool)
    if tokens.ndim != 3 or tokens.shape[1] != 4 or mask.shape != (4,) or mask.sum().item() != 2:
        raise ValueError("Expected four patches with exactly two masked")
    return tokens.masked_fill(mask[None, :, None], 0)


class AbsoluteWindowDataset(Dataset):
    """Step is always one, including threshold scoring. No split crossing."""
    def __init__(self, values, absolute_start, log_store):
        self.values = np.asarray(values, dtype=np.float32)
        self.absolute_start = int(absolute_start)
        self.log_store = log_store
        if self.values.ndim != 2 or len(self.values) < 24 or not np.isfinite(self.values).all():
            raise ValueError("Expected finite N x C, N >= 24")
        end = self.absolute_start + len(self.values)
        if not any(a <= self.absolute_start < end <= b for a, b in ((0, 4515), (4515, 5644), (5644, 7056))):
            raise ValueError("Window dataset outside an approved individual split")

    def __len__(self):
        return len(self.values) - 24 + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        absolute = self.absolute_start + index
        logs, present = self.log_store.window(absolute, 24)
        return self.values[index:index+24].copy(), logs, present, absolute


@torch.no_grad()
def score_series(model, dataset, *, device, batch_size=8, progress=None):
    """All six masks, masked squared errors, mean over masks then windows.

    Canonical start/mask/accumulation order is independent of global RNG.
    Coverage counts windows; mask_coverage counts scored masked observations.
    Error units are the external training-StandardScaler units, as historically.
    No DataLoader: even iterator construction must not consume global RNG.
    """
    if model.training or batch_size != 8:
        raise ValueError("Frozen eval model and batch_size=8 required")
    for name, value in (("seq_len", 24), ("patch_num", 4), ("patch_size", 6), ("stride", 6)):
        if getattr(model, name, None) != value:
            raise ValueError(f"Unsupported model {name}")
    n, c = dataset.values.shape
    totals = np.zeros((n, c), dtype=np.float64)
    coverage = np.zeros(n, dtype=np.int64)
    mask_coverage = np.zeros(n, dtype=np.int64)
    for offset in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(offset, min(offset + batch_size, len(dataset)))]
        x, log, present = [torch.as_tensor(np.stack([row[j] for row in rows]), device=device)
                           for j in range(3)]
        window_total = np.zeros((len(rows), 24, c), dtype=np.float64)
        window_counts = np.zeros(24, dtype=np.int64)
        for mask in MASK_BANK:
            reconstructed = model(x, log, present, fixed_patch_mask=mask, deterministic_gate=True)[0]
            if reconstructed.shape != x.shape or not torch.isfinite(reconstructed).all():
                raise ValueError("Nonfinite or misaligned reconstruction")
            point_mask = np.repeat(mask, 6)
            # Index first: visible errors cannot affect the sum (even huge ones).
            selected = torch.as_tensor(np.flatnonzero(point_mask), device=device)
            error = (reconstructed.index_select(1, selected) - x.index_select(1, selected)).square()
            window_total[:, point_mask] += error.cpu().numpy().astype(np.float64)
            window_counts[point_mask] += 1
        if not np.all(window_counts == 3):
            raise AssertionError("Six-mask bank must cover every point exactly three times")
        for j, row in enumerate(rows):
            start = row[3] - dataset.absolute_start
            totals[start:start+24] += window_total[j] / window_counts[:, None]
            coverage[start:start+24] += 1
            mask_coverage[start:start+24] += window_counts
        if progress:
            progress(offset + len(rows), len(dataset))
    if not np.all(coverage > 0):
        raise AssertionError("Uncovered point; padding is forbidden")
    errors = totals / coverage[:, None]
    scores = errors.mean(axis=1)
    if scores.shape != (n,) or not np.isfinite(scores).all():
        raise AssertionError("Invalid full-length scores")
    return dict(score=scores, channel_error=errors, coverage=coverage, mask_coverage=mask_coverage,
                absolute_index=np.arange(dataset.absolute_start, dataset.absolute_start+n))


def threshold_candidates(training_scores, validation_scores, ratios):
    """Same 61 percentiles, now over unique-point scores from score_series()."""
    if training_scores.shape != (4515,) or validation_scores.shape != (1412,):
        raise ValueError("Threshold pool must contain the approved point-level splits")
    expected = np.r_[np.arange(.5, 10.5, .5), np.arange(11, 52)]
    if not np.array_equal(np.asarray(ratios), expected):
        raise ValueError("Cannot change threshold search candidates")
    pool = np.concatenate([training_scores, validation_scores])
    if not np.isfinite(pool).all():
        raise ValueError("Nonfinite threshold scores")
    thresholds = np.percentile(pool, 100 - expected, method="linear")
    return thresholds, validation_scores[None, :] > thresholds[:, None]


def score_threshold_and_prediction(model, training_dataset, validation_dataset, *, device, progress=None):
    training = score_series(model, training_dataset, device=device, progress=progress)
    validation = score_series(model, validation_dataset, device=device, progress=progress)
    ratios = np.r_[np.arange(.5, 10.5, .5), np.arange(11, 52)]
    thresholds, predictions = threshold_candidates(training["score"], validation["score"], ratios)
    return training, validation, ratios, thresholds, predictions
