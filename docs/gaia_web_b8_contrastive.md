# GAIA web-only log-v2 contrastive experiment

This opt-in branch preserves the log-v2 backup at commit
`f55d07e94b49b056d2661b27ccaf69d5c87834c7` (`codex/gaia-cmc-loss`).
It is a MindTS improvement, **not a reproduction of MAD-CMC**. Only
GAIA_webservice1 and GAIA_webservice2 participate; webservice corresponds to the
paper's D2 service group, not D1. Splits/preprocessing still follow this project.
Do not compare the two-instance macro average with the historical ten-instance average.

## Objective and boundaries

`L = MSE + lamda2 * IB + 0.1 * (intra_metric + intra_log)/2 + 0.1 * inter`.

- Preserve the numeric encoder, log-v2 minute encoder, fusion, bottleneck,
  reconstruction head, and reconstruction anomaly scores.
- Add separate metric/log projection heads (hidden 128, output 64).
- Numeric tokens: mean across channels, retain the patch dimension, then flatten
  **only for the auxiliary head**. Main-path channels are not averaged away.
- Log tokens: retain 24 minute positions, flatten for the auxiliary head.
- Two feature-dropout views (p=0.05) share each modality's projection head. These
  are lightweight feature-space views, not two independently encoded raw inputs.
- Symmetric metric-metric and log-log InfoNCE; symmetric metric-log InfoNCE across
  matched views. Temperature 0.2; mean reductions over eligible anchors/directions.
- Positive: same training window's paired view / aligned modality. Negative:
  another window from the same service, absolute start distance >=24.
- Missing-log windows participate in numeric contrast but not log/cross contrast.
  Anchors without at least one legal negative contribute zero (recorded in logs).
- No labels, clustering, pseudo-labels, memory queue, or test examples form pairs.
- Nonoverlap does NOT guarantee a different system state. False negatives remain
  a limitation, and batch=8 provides at most seven candidates per direction.

Fixed seed 2021; window24/stride1; gradient training `[0,4515)`, early-stop
`[4515,5644)`, proposal validation `[5644,7056)`. No final-test evaluation.
Scaler fit uses gradient training only. Batch8, final partial batch retained,
3 epochs, Adam lr=1e-4; remaining baseline settings come from the frozen config.

The new trainer explicitly uses fixed-RNG reconstruction MSE for early stopping,
**not the old combined validation loss**. Thus also run the provided
`--objective legacy_control` before attributing gains solely to contrastive loss.
That control shares new heads, RNG consumption, batching, early-stop, and checkpoint
plumbing, but uses the historical summed CLIP objective. It is not a rerun of the
old batch1 experiment. The inherited learning-rate update function is unchanged.

The new mode calls the same frozen LLM backbone without its unused vocabulary
projection. This avoids constructing unused logits, retaining the same backbone
configuration, input batch, dtype, and RNG. The old entry point remains unchanged.
A tiny Qwen CPU test compares hidden states and RNG exactly; this is not a
batch8/4090 capacity guarantee. Other intermediate tensors can still be large.

## Metrics remain frozen

The new runner delegates scoring, percentile candidates, threshold selection,
and eight metrics to the unchanged log-v2 F1 runner. Select the threshold of
maximum **ordinary point F1**, not maximum affiliation_f; preserve tie handling.
Record affiliation_f, PR, RC, F1, AUC, AP, V-ROC, V-PR and the actual threshold.
`mean_best_f1` is ordinary F1 averaged equally over exactly two web instances.
The last nonempty successful runner line is `{"mean_best_f1": 0.0000}`.

Historical random masks/gates in inference are retained; model.eval does not
disable them. Historical tail20 zero-padding is also retained, not repaired here.
Neither should be mistaken for a desirable final protocol; changing them requires
a separate approved experiment and new baseline.

## Checkpoints / replay

Each external run directory stores resources/config/tokenizer, code and approved
input-prefix fingerprints, environment versions, Git commit, pip freeze,
per-service results, arrays, thresholds and checkpoints. No raw logs or checkpoint
weights are pushed to GitHub. Cache/dataset files must be backed up separately.

- `last.pt`: complete model (including frozen LLM), Adam state, learning-rate
  state, scaler, sampler permutation/cursor, epoch/step/patience, best weights,
  Python/NumPy/CPU/CUDA RNG. Atomic first-step, every50 steps, and epoch-end writes.
- `best.pt`: selected epoch weights and resumable optimizer/progress/RNG.
- `evaluation_start.pt`: selected model and **exact pre-scoring RNG**, required
  for replaying historical random inference. Best weights alone are insufficient.
- `training.jsonl`: per-step losses, valid anchor counts, candidate counts.

Interrupted training resumes from the last complete checkpoint, possibly redoing
up to49 unsaved updates. It never claims to recover an unpersisted in-memory step.
Resume/replay use a fresh output directory; never overwrite the source run.
Completed services are replayed, interrupted services resumed, and a service
that never started is trained fresh. Missing checkpoints in a started service fail
closed. A full checkpoint includes large frozen weights: budget substantial disk
and save time. Atomic replacement temporarily needs extra disk space.

Replay verifies saved arrays bit-for-bit when an original result exists. Strict
code/config/data/resource/environment identity mismatch fails, not silently
reseeded/retrained. Matching versions/hardware is necessary, not a universal
cross-platform bitwise guarantee. Never load an untrusted pickle checkpoint.

The preflight guard checks historical frozen **code/config**, not complete raw
dataset hashes (which would read test suffixes). Separate provenance hashes only
the approved 7056-point numeric prefix and the v2 cache manifest; LogV2Store
validates the selected service cache/vocabulary. This is not the historical
full-dataset guard. Use no whole-dataset hash scan in this workflow.

## Commands (server repository root)

CPU checks, no GAIA training:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tests/test_mindts_web_contrastive.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python tests/test_mindts_web_resume.py
```

Read-only preflight (default mode):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/autoresearch/gaia_web_b8_runner.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2
```

Only when GPU capacity is available, bounded one-step forward/backward smoke test
(not a full experiment, does update a disposable in-memory model):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/smoke_mindts_web_b8.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2 \
  --service GAIA_webservice1
```

Full run requires explicit authorization and a clean committed checkout. Example
only, **not automatically executed**:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/autoresearch/gaia_web_b8_runner.py \
  --cache-dir /home/xuke/dyao/autoresearch-tools/gaia/log_features_v2/cache-20260907-v2 \
  --output-dir /home/xuke/dyao/autoresearch-tools/gaia/runs/web-b8-dual-RUN_ID \
  --objective dual --run-validation
```

For a matched control use `--objective legacy_control` and a different fresh
output directory. To resume add `--resume-from ORIGINAL_RUN` to the same-objective
training command with a fresh output directory. For no-training replay replace
`--run-validation` by `--replay-from ORIGINAL_RUN`. Retain both service results;
never omit a failed service from the mean. No test-set command is provided.
