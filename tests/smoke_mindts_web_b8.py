"""Explicit bounded training-only resource probe; no validation metrics or test access."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--service", choices=("GAIA_webservice1", "GAIA_webservice2"), required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from scripts.autoresearch.gaia_web_b8_runner import configure
    from ts_benchmark.baselines.MindTS.MindTS import MINDTSConfig, Bottleneck_loss
    from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
    from ts_benchmark.baselines.MindTS.cmc_log_data import LogV2Store, load_numeric_prefix
    from ts_benchmark.baselines.MindTS.contrastive_loss import dual_contrast
    from sklearn.preprocessing import StandardScaler
    candidate, protocol, baseline = configure()
    service = next(s for s in protocol["services"] if s["name"] == args.service)
    store = LogV2Store(args.cache_dir, args.service)
    frame = load_numeric_prefix(ROOT / "dataset/anomaly_detect/data" / service["series"], service["channels"])
    # Fit exclusively on original gradient-training range, then use eight fixed nonoverlapping windows.
    scaler = StandardScaler().fit(frame.iloc[:4515].values)
    starts = np.arange(8) * 32
    x = np.stack([scaler.transform(frame.iloc[s:s+24].values) for s in starts])
    log, present = zip(*(store.window(int(s), 24) for s in starts))
    params = dict(baseline)
    params.update({k: candidate[k] for k in ("batch_size", "contrastive_enabled", "contrast_hidden", "contrast_dim", "contrast_dropout")})
    params.update(log_input_mode="v2", log_semantic_dim=store.semantic_dim, log_count_dim=store.count_dim,
                  enc_in_time=service["channels"], enc_in=service["channels"])
    torch.manual_seed(2021)
    torch.cuda.reset_peak_memory_stats()
    model = MINDTSModel(MINDTSConfig(**params)).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline["lr"])
    x, log, present = (torch.tensor(np.asarray(v), dtype=torch.float32, device="cuda") for v in (x, log, present))
    start = time.monotonic()
    out, _, _, mask, views = model(x, log, present, return_contrast=True)
    terms = dual_contrast(views, torch.tensor(starts, device="cuda"), present)
    loss = (out-x).square().mean() + Bottleneck_loss(mask, baseline["r"], baseline["lamda"])
    loss = loss + .1*(terms["intra_metric"]+terms["intra_log"])/2 + .1*terms["inter"]
    loss.backward()
    for module in (model.time_patch_encoder, model.minute_log_encoder, model.metric_contrast_head, model.log_contrast_head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters())
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({"batch8_probe": "ok", "service": args.service, "loss": float(loss.detach()),
                      "elapsed_seconds": time.monotonic()-start,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
                      "optimizer_steps": 1, "validation_evaluations": 0, "test_used": False}), flush=True)


if __name__ == "__main__":
    main()
