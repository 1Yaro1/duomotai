"""Label-free, in-batch window contrast. No queue, clustering or scoring changes."""
import torch
from torch import nn
from torch.nn import functional as F


class WindowContrastHead(nn.Module):
    def __init__(self, tokens, width, hidden=128, projection=64, dropout=0.05):
        super().__init__()
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(tokens * width, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection))

    def forward(self, tokens):
        def view():
            x = F.dropout(tokens, p=self.dropout, training=self.training)
            return F.normalize(self.network(x.flatten(1)), dim=-1)
        return view(), view()


def negative_mask(starts, window=24):
    """One service per batch; starts are absolute training indices, not batch IDs."""
    if starts.ndim != 1 or starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integer window starts")
    if window != 24 or torch.any(starts < 0) or torch.any(starts + window > 4515):
        raise ValueError("Contrast accepts gradient-training windows only")
    return (starts[:, None] - starts[None, :]).abs() >= window


def directional_nce(query, positive, negatives, *, temperature=0.2, min_negatives=1):
    """Diagonal positives; off-diagonal allowed negatives. Both sides get gradients."""
    if temperature <= 0 or min_negatives < 1:
        raise ValueError("Invalid contrast settings")
    if query.ndim != 2 or query.shape != positive.shape:
        raise ValueError("Expected paired [B,D] embeddings")
    b = query.shape[0]
    if negatives.shape != (b, b) or negatives.dtype != torch.bool or negatives.diag().any():
        raise ValueError("Invalid negative mask / self-negative")
    if not torch.isfinite(query).all() or not torch.isfinite(positive).all():
        raise ValueError("Non-finite embeddings")
    valid = negatives.sum(1) >= min_negatives
    count = int(valid.sum().item())
    if count == 0:
        return (query.sum() + positive.sum()) * 0, 0
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    logits = (q @ p.T) / temperature
    allowed = negatives | torch.eye(b, dtype=torch.bool, device=query.device)
    logits = logits.masked_fill(~allowed, -torch.inf)
    target = torch.arange(b, device=query.device)
    return F.cross_entropy(logits[valid], target[valid], reduction="mean"), count


def dual_contrast(views, starts, present, temperature=0.2, min_negatives=1):
    """Feature-space views, not two full LLM forwards; all reductions are means."""
    ma, mb, la, lb = (views[k] for k in ("ma", "mb", "la", "lb"))
    if present.shape != (ma.shape[0], 24) or not torch.all((present == 0) | (present == 1)):
        raise ValueError("Expected binary [B,24] log presence")
    mask_m = negative_mask(starts)
    available = present.bool().any(1)
    mask_l = mask_m & available[:, None] & available[None, :]
    def pair(a, b, mask):
        return directional_nce(a, b, mask, temperature=temperature, min_negatives=min_negatives)
    m1, nm = pair(ma, mb, mask_m)
    m2, _ = pair(mb, ma, mask_m)
    l1, nl = pair(la, lb, mask_l)
    l2, _ = pair(lb, la, mask_l)
    cross = [pair(a, b, mask_l)[0] for a, b in ((ma, la), (la, ma), (mb, lb), (lb, mb))]
    return {"intra_metric": (m1 + m2) / 2, "intra_log": (l1 + l2) / 2,
            "inter": sum(cross) / 4,
            "valid_metric": nm, "valid_log": nl, "valid_inter": nl,
            "negative_metric_mean": float(mask_m.sum(1).float().mean().item()),
            "negative_log_mean": float(mask_l.sum(1).float().mean().item())}
