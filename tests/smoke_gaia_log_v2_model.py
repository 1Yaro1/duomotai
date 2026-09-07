"""Explicit synthetic interface smoke. Loads frozen local models, never GAIA."""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-path", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("log_v2_prep", ROOT / "scripts/autoresearch/prepare_gaia_log_features_v2.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    torch.manual_seed(2021)
    encoder = prep.FrozenBert(args.bert_path, device="cpu")
    vec = encoder(["INFO request completed", "ERROR connection refused", "word " * 700 + " end"])
    assert vec.shape == (3, 768) and np.isfinite(vec).all()
    assert all(not p.requires_grad for p in encoder.model.parameters())
    print("Frozen local BERT: 3 synthetic templates including >512 tokens; PASS", flush=True)
    del encoder

    from ts_benchmark.baselines.MindTS.MindTS import MINDTSConfig
    from ts_benchmark.baselines.MindTS.models.MindTS_model import MINDTSModel
    config = MINDTSConfig(seq_len=24, enc_in_time=2, enc_in=2, batch_size=1, d_ff=8,
                         log_input_mode="v2", log_semantic_dim=768, log_count_dim=3,
                         mask_ratio=0.4, patch_size=6, stride=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MINDTSModel(config).to(device)
    model.eval()
    metric = torch.randn(1, 24, 2, device=device)
    features = torch.zeros(1, 24, 771, device=device)
    features[0, :3, :768] = torch.tensor(vec, device=device)
    present = torch.zeros(1, 24, device=device)
    present[:, :3] = 1
    output, a, b, mask = model(metric.clone(), features, present)
    assert output.shape == metric.shape
    assert a.shape == b.shape == (2, 4, 4)
    assert all(torch.isfinite(x).all() for x in (output, a, b, mask))
    # No optimizer step, no checkpoint saved, no anomaly metric calculated.
    (output.square().mean() + a.square().mean() * 1e-4).backward()
    grads = [p.grad for p in model.minute_log_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    print("MINDTS log-v2: synthetic 24-minute forward/backward, no optimizer step; PASS", flush=True)
    print("GAIA_DATA_READ=0; TRAINING_RUNS=0; TEST_EVALUATIONS=0", flush=True)


if __name__ == "__main__":
    main()
