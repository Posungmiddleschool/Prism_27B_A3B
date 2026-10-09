# PRISM — a 27B model that reads 3B parameters per token

PRISM is an independent research project. It takes the dense, Apache-2.0 **Qwen3.8-27B** and tries to turn it,
after training, into a sparse model that touches **at most 3.0B of its 27B parameters for each token** — a
"27B-A3B" — without retraining the base weights.

The goal is deliberately hard: **beat a dense 9B model on reasoning benchmarks while running several times
faster than the 27B original.** That goal has **not** been reached. This page reports where the project
actually stands, including the experiments that failed and one measurement error we found in our own work.

## The idea

Most compression work shrinks each layer of a dense model. PRISM treats the dense model differently:

> The 27B weights stay frozen as **read-only memory**. For every token, a small learned component decides
> which 3B of them to read. The skill of the sparse model lives in *what it chooses to read*.

Because the base weights are never changed, the dense teacher lives inside the same weights as the student
and can be switched on at any time for distillation or comparison.

**Counting rule (strict since v16).** Every parameter a token's forward pass reads is counted by a census —
exact rows, routers, predictors, low-rank corrections and the output head included — and the limit holds for
**every token**, not on average. Current census: **2.998B** per token, measured maximum over held-out tokens
**2.999B**.

## Correction: the v13–v15 numbers were measured with a dense output head

The v16 audit found two places where the forward pass did not match the census:

1. **Output head.** The census charged a small "head funnel" (rank-256 proxy + 4,096 exact vocabulary rows,
   0.086B), but the forward pass still used the full dense head (1.27B). Every PPL and KL number for v13–v15
   was measured with that dense head. The earlier claim "head funnel — measured loss: none" was wrong.
2. **Memory writes.** The census charged 4 exact memory writes per token in each Gated-DeltaNet layer. That was
   the *average*: individual tokens wrote anywhere between 0 and 36 of 48 heads.

Measured on the same model and the same held-out blocks, closing both:

| | text KL | reasoning KL |
|---|---|---|
| as reported before (dense head, average writes) | 1.78 | 0.62 |
| strict writes (exactly 4 per token) | 1.78 | 0.63 |
| + strict head (the head the census counts) | 2.88 | 2.61 |

So the strict writes cost almost nothing, while the dense head was hiding most of the reasoning error. From
v16 on, every reported number is measured on the model as it is counted.

## Mechanisms

| Name | Where | What it does |
|---|---|---|
| **Neuron lens** | FFN | A rank-128 predictor estimates every neuron's activation; ~480 neurons per layer and token are computed exactly with the original weights; the rest contribute their mean plus a closed-form low-rank "shadow". |
| **Shadow execution** | attention projections | `y = S(x) + selected · (exact(x) − S(x))`: a cheap low-rank estimate everywhere, exact rows only where a router selects them. |
| **Strict memory writes (MoRM)** | Gated-DeltaNet layers | Exactly 4 of 48 value heads write exactly per token; a small critic picks them. |
| **Strict head + LIFT** (v16) | output layer | A rank-256 proxy scores the whole vocabulary; the top 5,888 rows are computed exactly. LIFT trains the proxy only to lift the teacher's tokens above the shortlist line, instead of regressing all 248k logits. |
| **Wallet** (v16) | every selectable site | Each token gets its own budget. It may buy more neurons or rows at one site by selling at another; its balance can never go below zero, so every token stays under the ceiling. Prices are set from gradient probes. |
| **Own neurons** (v16) | FFN | The 256 neurons per layer that carry the most output get trainable copies of their weights. |
| **Fork loss** (v16) | loss | Full penalty only where the student leaves every token the teacher finds acceptable. |
| **Stream anchor** | whole model | The student's residual stream is pulled toward the teacher's at every 4th layer during distillation. |

## Results

Held-out text the model was not trained on. Dense teacher PPL: **6.53**.

**v9–v15** (dense output head, average memory writes — see the correction above):

| Version | What changed | Held-out PPL | KL to teacher |
|---|---|---|---|
| v9 | first sparse surgery | ~200,000 | — |
| v10 | shadow execution, 400 distillation steps | 160.2 | 3.43 |
| v11 | closed-form stream realignment | **rolled back** (1,576) | 5.73 |
| v12 | neuron lens, no gradient steps | 156.0 | 3.37 |
| v13 | lens on every FFN + 324 healing steps | 41.5 | 2.17 |
| v14 | training-free budget market (~30 knobs, ±25%) | — | 2.15 |
| v15 | consequence head + attention shadows, 322 steps | 31.35 | — |

**v16.2 — under the strict law** (every token ≤ 3.0B, the head that is counted is the head that runs):

| | before v16 training | after (310 steps, 201 min, one A100 80GB) |
|---|---|---|
| held-out PPL | 143.32 | **33.40** |
| text KL / first-token agreement | 3.26 / 43.8% | **1.96** / 47.8% |
| reasoning KL / first-token agreement | 1.90 / 70.9% | **0.58** / 78.7% |
| decisive positions (teacher ≥ 0.9 sure) matched | 83.7% | 91.1% |
| 12 decisive positions in a row without a mistake | 23.2% | **45.8%** |
| teacher probability inside the head shortlist | 85.7% | 97.2% |
| one-step arithmetic (20 problems) | 1/20 (v15) | **2/20** |

PPL 33.40 under the strict count is close to the 31.35 that v15 reported *with* the uncounted dense head.

**What mattered** (one part reset to its pre-training state, everything else kept):

| Reset part | text KL | reasoning KL |
|---|---|---|
| nothing (final model) | 1.964 | 0.577 |
| strict head proxy | +0.907 | +0.691 |
| consequence head | +0.045 | +0.015 |
| FFN lens shadows | +0.017 | +0.022 |
| wallet (every token buys the fixed quotas) | +0.011 | +0.021 |
| own neurons | +0.009 | +0.009 |
| attention shadows | −0.030 | +0.013 |

**The wallet in use.** On one held-out block, tokens touched between 2.69B and 2.999B (mean 2.84B). The number
of exact FFN neurons a token bought ranged from 268 to 723 per layer. Tokens sold attention-side rows and
memory writes and bought FFN neurons.

## What did not work

- **v11 — stream realignment.** Matching intermediate states lowered every per-layer error and made the model
  far worse (PPL 160 → 1,576). Rolled back.
- **v12 — lens without healing.** The rest of the model had been trained around the broken FFNs.
- **v16.1 — NaN at step 2.** The memory-write mask used a straight-through gradient in score space,
  `d/ds sigmoid((e^s − e^t)/τ)`, which grows with `e^s`. On tokens with large critic scores it reached ~10¹⁶,
  overflowed in the backward pass and spread NaN to every layer below. Fixed by moving the mask to log-score
  space (the choice of heads is unchanged; the gradient is bounded by 1/4τ).
- **Wallet price rule.** Prices rise when tokens *want* more than the free budget. The demand comes from the few
  tokens whose wallets are already empty, so raising prices mostly makes the other tokens sell: about 157M per
  token is left unspent. To be fixed in v17.

## Current limitations

- **Quality is far from the goal.** PPL 33.4 against 6.53 for the original.
- **Reasoning:** one-step arithmetic 2/20 (teacher 20/20). Wrong answers are now near misses (67+25 → 91,
  50−16 → 32); multiplication is still wrong. GSM8K is not run until arithmetic passes 30%.
- **No benchmark numbers** (GPQA Diamond, HMMT, MMLU-Pro) yet.
- **Slower than the original today.** The sparse path is computed through masks over dense operations:
  decode 1.2 tokens/s against 5.5 for the dense model in the same loop. No sparse kernel yet.
- **Long context is untested.** Training and evaluation use 1,024-token blocks only.
- **Small training set.** 1,408 sequences (mixed web text, Korean Wikipedia, code, chat, GSM8K-train,
  arithmetic, and reasoning traces), seen about twice.
- 932.5M parameters (3.3%) are trainable; the base weights stay frozen by design.

## Gates

1. one-step arithmetic: more than half correct — **2/20 now**
2. GSM8K: more than half correct
3. GPQA Diamond above a dense 4B model (76.2)
4. GPQA Diamond above a dense 9B model (81.7) — the goal. The 27B original scores 89.2.

None has been passed yet.

## Next (v17)

- **Learning from its own answers.** The student writes arithmetic answers itself and the teacher grades every
  token it wrote, so it learns at the places where it actually goes wrong.
- **Fix the wallet price rule** so tokens spend the whole budget.
- **Fresh data every step.** The teacher already runs in every training step for the stream anchor; its next-token
  distribution is taken there instead of from a fixed cache, so no sequence is seen twice.
- **Long-context curve.** KL at 1K / 4K / 16K / 32K positions, reported every version.
- **Per-source KL** in the training log.

## Reproducing

Each version is a self-contained notebook that restores the previous version's saved state.

- Hardware: one GPU with 80 GB (A100 80GB, H100, H200).
- v16.2: about 1 hour of setup and audit + 200 minutes of training on one A100 80GB.

## Base model and license

Base model: Qwen3.8-27B by Alibaba Cloud (Apache-2.0). Reference scores for the dense 4B / 9B / 27B models
are taken from their public model cards.

Code in this repository: Apache-2.0.
