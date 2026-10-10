# PRISM — a 27B model that reads 3B parameters per token

PRISM is an independent research project. It takes the dense, Apache-2.0 **Qwen3.8-27B** and tries to turn it,
after training, into a sparse model that touches **at most 3.0B of its 27B parameters for each token** — a
"27B-A3B" — without retraining the base weights.

The goal is deliberately hard: **beat a dense 9B model on reasoning benchmarks while running several times
faster than the 27B original.** That goal has **not** been reached. This page reports where the project
actually stands, including the experiments that failed and the measurement errors we found in our own work.

**Latest: v17** — under the strict per-token count, held-out perplexity **28.9** (dense teacher 6.5). This is
better than the 31.4 that v15 reported while it was still, by mistake, using an uncounted dense output head.
The student's accuracy on arithmetic answers it writes itself rose from 7% to 13% during training.

## The idea

Most compression work shrinks each layer of a dense model. PRISM treats the dense model differently:

> The 27B weights stay frozen as **read-only memory**. For every token, a small learned component decides
> which 3B of them to read. The skill of the sparse model lives in *what it chooses to read*.

Because the base weights are never changed, the dense teacher lives inside the same weights as the student
and can be switched on at any time for distillation or comparison.

**Counting rule (strict since v16).** Every parameter a token's forward pass reads is counted by a census —
exact rows, routers, predictors, low-rank corrections and the output head included — and the limit holds for
**every token**, not on average. Current census: **2.998B** at the quotas; the largest per-token reading measured
on held-out text is 2.9996B.

## Results

Held-out text the model was not trained on. Dense teacher PPL: **6.53**.

| Version | What changed | Held-out PPL | One-step arithmetic |
|---|---|---|---|
| v10 | shadow execution, 400 distillation steps | 160.2 | 0/20 |
| v13 | neuron lens on every FFN + healing | 41.5 * | 0/20 |
| v15 | consequence head, attention shadows | 31.35 * | 1/20 |
| v16.2 | **strict law**, strict head + LIFT, wallet | 33.40 | 2/20 |
| **v17** | in-step teacher, full-vocabulary KL, DRIFT, LOOP, wallet fix | **28.94** | 3/20 |

\* measured with the uncounted dense output head (see the correction below). From v16.2 on, every number is measured
on the model exactly as it is counted. Full tables: [RESULTS.md](RESULTS.md). Run-by-run notes:
[WORKLOG_v14-v16.md](WORKLOG_v14-v16.md), [WORKLOG_v17.md](WORKLOG_v17.md).

**v17 in numbers** (small held-out eval, v16.2 → v17): text KL 1.862 → 1.699; positions where the teacher is ≥ 90%
sure, matched 91.4% → 91.9%; held-out PPL 33.4 → 28.9; the student's own LOOP answers 7% → 13% correct; mean
parameters per token 2.84B → 2.98B (the wallet now spends its budget).

## Correction: the v13–v15 numbers were measured with a dense output head

The v16 audit found two places where the forward pass did not match the census: the output head ran dense
(1.27B uncounted), and memory writes were counted as an average (tokens wrote 0–36 of 48 heads). Both were closed in
v16. On the same blocks, the honest head raised text KL from 1.78 to 2.88 and reasoning KL from 0.62 to 2.61. v16.2
and v17 recovered all of it under the strict count.

## Mechanisms

| Name | Where | What it does |
|---|---|---|
| **Neuron lens** | FFN | Predicts every neuron's activation; ~480 neurons per layer and token are computed exactly, the rest contribute their mean plus a low-rank shadow. |
| **Shadow execution** | attention projections | A low-rank estimate everywhere, exact row-blocks only where a router selects them. |
| **Strict memory writes** | Gated-DeltaNet layers | Exactly 4 of 48 value heads write exactly per token; a critic picks them (straight-through in log-score space). |
| **Strict head + LIFT** | output layer | A rank-256 proxy scores the vocabulary; ~6,000 rows are exact. LIFT trains the proxy to lift the teacher's tokens over the shortlist line. |
| **Wallet** | every selectable site | Each token gets its own budget and may buy more items at one site by selling at another; it can never go below zero. Prices come from gradient probes; the price level follows the unspent budget. |
| **In-step teacher, full KL** | training | The dense teacher labels each batch inside the training step; the loss is the KL to its whole 248k-token distribution. |
| **DRIFT** | training | The student's own wrong predictions are written into the text it reads; the teacher shows how to go on from them. |
| **LOOP** | training | The student writes its own answers to fresh problems; the teacher grades every token it wrote. |
| **Fuse** | training | A non-finite loss or gradient never reaches the weights; the deepest affected tensor is named. |

Details and equations: [docs/METHODS.md](docs/METHODS.md).

## What did not work

- **v11 — stream realignment.** Matching intermediate states made the model far worse (PPL 160 → 1,576). Rolled back.
- **v12 — lens without healing.** The rest of the model had been trained around the broken FFNs.
- **v16.1 — NaN at step 2.** The memory-write straight-through gradient grew with `e^score` and overflowed. Fixed by
  moving it to log-score space.
- **v16.2 / v17 — zero reasoning traces.** The trace loader merged two incompatible dataset folders and read the
  wrong column; every trace was silently dropped. Fixed for v18, which now stops with a banner when this happens.
- **v17 — DRIFT on arithmetic.** Rewriting digits in problem rows dropped "12 decisive positions in a row" from 46% to
  25% at step 50 (it recovered to 42% by step 200). v18 keeps DRIFT out of problem rows.
- **v17 — the wallet sold the memory.** Tokens kept 1.0 of 4 exact memory writes and 0.4 of 1.8 query row-blocks to
  buy FFN neurons. v18 lets a token sell at most one of each.

## Current limitations

- **Quality is far from the goal.** PPL 28.9 against 6.53; one-step arithmetic 3/20 (teacher 20/20).
- **No benchmark numbers yet.** GSM8K, MMLU-Pro and the long-context curve are measured from v18 on; GPQA is not run.
- **Slower than the original today.** Sparse paths are computed through masks over dense operations (decode 1.2 tok/s
  vs 5.5 for the dense model in the same loop). No sparse kernel yet.
- **Long context untested.** All training so far used 1,024-token rows; v18 is the first with 2,048-token rows.
- **Small compute.** One A100 80GB, about 3.5 hours of training per version.

## Gates

1. one-step arithmetic: more than half correct — **3/20 now**
2. GSM8K: more than half correct — measured from v18
3. GPQA Diamond above a dense 4B model (76.2)
4. GPQA Diamond above a dense 9B model (81.7) — the goal. The 27B original scores 89.2.

## Repository

| File | What |
|---|---|
| `PRISM_v10 … v18_Qwen3.8-27B_A3B.ipynb` | one self-contained notebook per version; each restores the previous version's saved state |
| [RESULTS.md](RESULTS.md), [results.csv](results.csv) | every measured number, per version |
| [WORKLOG_v14-v16.md](WORKLOG_v14-v16.md), [WORKLOG_v17.md](WORKLOG_v17.md) | what was changed, what failed, why |
| [docs/METHODS.md](docs/METHODS.md) | the mechanisms, with equations |

## Base model and license

Base model: Qwen3.8-27B by Alibaba Cloud (Apache-2.0). Reference scores for the dense 4B / 9B / 27B models are taken
from their public model cards. Code in this repository: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
