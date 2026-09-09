# D1 dbservice batch8 experiment

Active target: **GAIA_dbservice1 and GAIA_dbservice2**, requested by the user
on 2026-09-09. New branch `codex/mindts-d1-b8-contrastive`; old web branch/config
and original log-v2 backup remain available. No dataset files are renamed,
replaced, removed, or regenerated. This uses the requested D1 service group but
the project's approved splits and evaluator, **not a strict paper reproduction**.

## Entry points

- `config/autoresearch/gaia_d1_b8_contrastive.json`: exactly two db services,
  seed2021, batch8, window24/stride1, 3epochs, dual contrast, fixed-RNG MSE early stop.
- `scripts/autoresearch/gaia_d1_b8_runner.py`: independent D1 command, result
  identity, aggregation, resume and replay gates. Uses the unchanged F1 evaluator.
- `ts_benchmark/baselines/MindTSD1.py`: inherits tested persistent-checkpoint
  training machinery, with its own db-only allowlist. Web trainer still web-only.
- `tests/smoke_mindts_d1_b8.py`: one disposable training-batch update, no metrics.

Gradient fit `[0,4515)`, early stop `[4515,5644)`, proposal validation
`[5644,7056)`. No final test. Only ordinary F1 selects threshold; report
affiliation_f, PR, RC, F1, AUC, AP, V-ROC, V-PR. `mean_best_f1` is the equal-weight
average of ordinary Best F1 over **both db instances**. Never mix web results in.
Paired `legacy_control` retains the same batch, early stop and memory optimization,
changing only the loss objective; old batch1/web results are not matched controls.

## Memory-only optimization

D1 sets `llm_memory_efficient=true`; web and legacy modes default false.
The frozen prompt LLM returns only its final hidden representation, without
retaining every hidden layer, attention matrix or autoregressive KV cache.
No prompt batching/chunking, mixed precision, quantization or seed change.
The numeric encoder, log encoder, fusion, contrastive batch and reconstruction
are untouched. The previously verified prompt-only CPU median remains enabled.

The original Qwen2 SDPA module fell back to eager attention when attention outputs
were requested. Simply suppressing attention outputs would switch kernels.
Therefore D1 explicitly constructs **eager attention**, preserving that original
computation, while dropping unused outputs. Guard rejects an SDPA model in the
efficient helper. Full weights/state keys and checkpoint reconstruction remain
compatible with the configured eager architecture; replay uses strict identity.

New tests compare old SDPA-fallback against eager/no-retention on CPU/CUDA, in
train/eval, including padding/dropout, checking final hidden states and RNG.
`--real-llm-check` additionally checks actual six-layer pretrained weights with
synthetic prompt IDs, no GAIA data or training. A passing small comparison does
not by itself guarantee full batch8 capacity or all-dataset numerical equivalence.

## Commands (repository root)

Preflight only, no training:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/autoresearch/gaia_d1_b8_runner.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2
```

Bounded equivalence tests:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_mindts_d1_memory.py --real-llm-check
```

One service's disposable batch8 resource check (run separately for dbservice2):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/smoke_mindts_d1_b8.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2 \
  --service GAIA_dbservice1
```

Full training requires separate approval; this command is documentation only:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/autoresearch/gaia_d1_b8_runner.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2 \
  --output-dir /home/xuke/dyao/autoresearch-tools/gaia/runs/d1-b8-dual-RUN_ID \
  --objective dual --run-validation
```

Full runs persist last/best/evaluation_start checkpoints, optimizer, scaler,
sampler/RNG/early-stop state, model/tokenizer resources and source/data/environment
fingerprints; metrics and raw prediction arrays stay outside Git. For resume add
`--resume-from ORIGINAL_RUN`; for no-training replay replace `--run-validation`
with `--replay-from ORIGINAL_RUN`. Always use a fresh external output directory.
Web checkpoints/results fail D1 identity checks and must not be reused as D1 runs.

Historical random gates/masks and tail20 zero-padding remain unchanged. Model
quality and checkpoint score replay must be validated after authorized training.
