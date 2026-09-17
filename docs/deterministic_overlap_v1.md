# deterministic_overlap_v1 scoring protocol

This document records the opt-in deterministic point-level scoring protocol for
the existing MindTS checkpoints on `GAIA_dbservice1` and `GAIA_dbservice2`.
It is an evaluation-only protocol: it does not train a model and must not be
described as a model-architecture or training-objective improvement.

## Status and provenance

- Protocol name: `deterministic_overlap_v1`
- Git branch: `codex/db-deterministic-overlap-v1`
- Scoring implementation commit frozen by the evaluation contract:
  `b542ec8e66afd53ebf21be71dd2059815844dda6`
- Frozen contract SHA256:
  `ca2fbf7257705597488be0933e8d7bef548d94d606b3a01a8a7253badf07b3e6`
- Services: `GAIA_dbservice1`, `GAIA_dbservice2`
- Seed reference: `2021`
- Gradient-training range: `[0, 4515)`
- Early-stopping range: `[4515, 5644)`
- Proposal-validation range: `[5644, 7056)`
- Final test range `[7056, 10080)`: not used

The exact frozen machine-readable contract is checked into
`config/evaluation_contract_db_overlap_v1.json`. Run outputs, checkpoints,
prefix caches, raw arrays, and GAIA data remain outside Git.

## Why this protocol exists

The historical MindTS evaluation path had three important reproducibility
limitations:

1. evaluation still used random masking and a hard Gumbel gate;
2. threshold scoring and validation scoring did not provide a single,
   full-length point-level overlap-add series;
3. the historical path padded the final 20 validation points with zero scores.

`deterministic_overlap_v1` provides a separately named scoring path. Historical
results remain immutable and are not overwritten.

## Frozen scoring definition

For every approved split:

1. Generate all length-24 windows with `step=1` and retain each window's
   absolute start index.
2. Divide every window into four non-overlapping patches of six points.
3. Evaluate the six fixed pairs in `C(4,2)` order:
   `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)`.
4. For each mask, compute squared reconstruction error only at masked points.
   Visible-point error is excluded before accumulation.
5. Every point within a window is masked exactly three times. Average its three
   masked observations.
6. Overlap-add window errors to absolute time indices and average all windows
   covering the same point.
7. Average the per-channel errors to obtain one point score.
8. Require every point to have positive coverage. Do not pad the tail.

The model runs in evaluation mode with a deterministic soft gate. Evaluation
must not call `torch.rand` or `torch.nn.functional.gumbel_softmax`.

## Threshold selection

- Threshold calibration and final validation prediction must call the same
  `score_series()` implementation.
- The threshold pool contains 4515 unique training-point scores followed by
  1412 unique proposal-validation scores; labels are excluded from the pool.
- The 61 anomaly-ratio candidates are fixed at `0.5` through `10.0` in steps of
  `0.5`, followed by `11` through `51` in steps of `1`.
- Percentiles use NumPy's `linear` method.
- Prediction is `score > threshold`.
- Select the candidate with maximum ordinary point-F1 on proposal validation.
  If candidates tie, retain the first candidate in the frozen order.
- Point adjustment is forbidden.
- Services are aggregated using an equal-weight arithmetic macro average.

AP and VUS-PR are the primary model-comparison metrics. Precision, Recall, and
ordinary point-F1 are auxiliary metrics. Affiliation-F, AUC, and VUS-ROC are
reported additionally; Affiliation-F must not be called ordinary F1.

## Determinism and fail-fast requirements

- Use the existing `best.pt` only; no optimizer or training update is allowed.
- Score the complete training and validation ranges twice.
- Change global RNG state between passes instead of resetting a convenient seed.
- Compare every saved array byte-for-byte.
- Require batch size 8, float32 checkpoint/model computation, and float64 score
  accumulation.
- Require exactly one visible GPU and at least 21000 MiB free memory before a
  real checkpoint run.
- Do not silently change a parameter or retry a failed run.
- Do not catch an exception and return a default score.
- Results must be written to a new directory outside Git.
- Prefix-only cache identity and checkpoint/resource identity checks must pass
  before scoring.
- Raw GAIA CSV files must not be reopened by the scoring runner.

## Code locations

- `ts_benchmark/baselines/MindTS/deterministic_scoring.py`
  implements fixed masks, full-length overlap-add, threshold candidates, and the
  shared scoring function.
- `scripts/autoresearch/score_db_overlap_v1.py`
  validates checkpoint, scaler, resources, prefix cache, environment, and exact
  replay provenance, then performs two deterministic score passes when explicitly
  requested.
- `scripts/autoresearch/extract_db_prefix_once.py`
  provides the audited one-time prefix extraction path.
- `tests/test_deterministic_overlap_v1.py`
  contains the protocol unit tests.

## Required tests

The committed tests cover:

- bitwise deterministic repeated scoring after RNG state changes;
- output length exactly 1412 for proposal validation;
- positive coverage for every point;
- no tail padding;
- threshold and prediction sharing the same score function;
- visible positions not contributing to masked-only error;
- the six-mask bank covering every point three times;
- rejection of windows crossing a split boundary;
- exact checkpoint-config property restoration;
- rejection of incomplete checkpoint configuration;
- skipping an unparseable test suffix without decoding it during one-time prefix
  extraction.

Run from the repository root:

```bash
.venv/bin/python -m unittest tests.test_deterministic_overlap_v1
```

The scoring runner defaults to CPU identity checks. A real scoring pass requires
the explicit `--execute-score` flag and audited external input paths:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
  scripts/autoresearch/score_db_overlap_v1.py \
  --run /path/to/audited-run \
  --prefix /path/to/prefix-only-cache \
  --exact-report /path/to/exact-replay-report.json \
  --output /new/external/output-directory \
  --service GAIA_dbservice1 \
  --accept-historical-prefix-hash \
  --execute-score
```

Use a new process and output directory for the second service.

## Verified reference evidence

The frozen reference run completed two byte-identical passes for both DB
services. Each validation score had length 1412, all points had positive
coverage, and no tail padding was present. No training update or test data was
used.

At the ordinary point-F1 optimum under this scoring protocol:

| Service | Precision | Recall | Point-F1 | AP | VUS-PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| GAIA_dbservice1 | 0.5972 | 0.7818 | 0.6772 | 0.6116 | 0.7368 |
| GAIA_dbservice2 | 0.5263 | 0.9091 | 0.6667 | 0.5481 | 0.6360 |

These values are checkpoint rescoring results. They are not evidence that the
model improved through retraining.

## Known limitations

- Full-window instance normalization is retained.
- Existing full-window statistical prompts can access statistics from points
  masked for reconstruction.
- Scoring is offline and non-causal; future context can be used.
- The deterministic soft gate differs from the historical hard Gumbel gate.
- Therefore historical and deterministic results must be reported under their
  explicit protocol names rather than merged into one result series.
