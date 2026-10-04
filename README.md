# PRISM — a 27B model that reads 3B parameters per token

PRISM is an independent research project. It takes the dense, Apache-2.0 **Qwen3.8-27B** and tries to turn it,
after training, into a sparse model that touches **at most 3.0B of its 27B parameters for each token** — a
"27B-A3B" — without retraining the base weights.

The goal is deliberately hard: **beat a dense 9B model on reasoning benchmarks while running several times
faster than the 27B original.** That goal has **not** been reached. This page reports where the project
actually stands, including the experiments that failed.

## The idea

Most compression work shrinks each layer of a dense model. PRISM treats the dense model differently:

> The 27B weights stay frozen as **read-only memory**. For every token, a small learned component decides
> which 3B of them to read. The skill of the sparse model lives in *what it chooses to read*.

Because the base weights are never changed, the dense teacher lives inside the same weights as the student
and can be switched on at any time for distillation or comparison.

**Counting rule.** Every parameter a token's forward pass reads is counted by a census — exact rows, routers,
predictors and low-rank corrections included. The census of the current model is **2.989B** per token.

## Mechanisms

| Name | Where | What it does |
|---|---|---|
| **Neuron lens** | FFN (63% of the weights) | A rank-128 predictor estimates every neuron's activation for the current token; the ~520 neurons (3%) that matter most are computed exactly with the original weights; an unselected neuron contributes its mean; a closed-form low-rank "shadow" models what is left. |
| **Shadow execution** | attention projections | `y = S(x) + selected · (exact(x) − S(x))`: a cheap low-rank estimate everywhere, exact rows only where a router selects them. |
| **Write-sparse memories (MoRM)** | Gated-DeltaNet layers | Only a few value heads write exactly per token; a small critic picks them. |
| **Head funnel** | output layer | A low-rank proxy shortlists vocabulary rows; only those are computed exactly. Measured loss: none. |
| **Healing with a stream anchor** | whole model | A short distillation run in which the student's residual stream is also pulled toward the dense teacher's stream at every 4th layer. |

Per-token neuron selection is related to prior work on contextual sparsity. What is specific here is the
combination: selection at ~3% density, a mean-anchored closed-form shadow, and the frozen dense model serving
as both memory and teacher.

## Results so far

Held-out perplexity on text the model was not distilled on. Dense teacher: **6.53**.

| Version | What changed | Held-out PPL | KL to teacher (nats/token) |
|---|---|---|---|
| v9 | first sparse surgery | ~200,000 | — |
| v10 | shadow execution, 400 distillation steps (~7 h, one A100 80GB) | 160.2 | 3.43 |
| v11 | closed-form stream realignment | **rolled back** (1,576) | 5.73 |
| v12 | neuron lens, no gradient steps | 156.0 | 3.37 |
| v13 | lens on every FFN + 324 healing steps (160 min, one H200) | **41.5** | **2.17** |

**How well the lens reproduces a dense FFN** (relative error of the layer output on the same inputs; 0 = identical):

| Layer | v10 expert groups | Neuron lens | Oracle selection |
|---|---|---|---|
| 8 | 0.942 | 0.178 | 0.146 |
| 32 | 0.876 | 0.088 | 0.077 |
| 55 | 0.792 | 0.234 | 0.117 |

**Where the remaining distance comes from** (lens everywhere, before healing):

| Sparse part | KL to teacher |
|---|---|
| FFN only | 2.56 |
| attention only | 2.48 |
| output head only | 0.00 |
| everything | 3.62 |

## What did not work

- **v11 — stream realignment.** Re-solving each layer's correction so the student's stream matches the
  teacher's lowered every per-layer error and made the model far worse (PPL 160 → 1,576). It was rolled back.
  Lesson: matching intermediate states is not the same as matching outputs.
- **v12 — lens without healing.** The lens cut the local FFN error five-fold, yet switching it on everywhere
  made the model *worse* (KL 3.43 → 3.88): the rest of the model had been trained around the broken FFNs.
  Only layers 48–63 kept it, for a 2.5% gain.
- **Small local errors compound.** A 9–23% error per FFN, with everything else dense, still costs 2.56 nats.

## Current limitations

- **Quality is far from the goal.** PPL 41.5 against 6.53 for the original.
- **No reasoning ability yet.** On a probe of one-step arithmetic and GSM8K problems the dense teacher scores
  20/20 and 12/13; the sparse student scores **0** on both.
- **No benchmark numbers.** GPQA Diamond, HMMT and similar have not been run on the student.
- **It is slower than the original today.** The research code computes the sparse path through masks over
  dense operations: 4.7 tokens/s against 12.9 for the dense model in the same loop. A real sparse kernel has
  not been written, and the 3× speed target is an untested hypothesis.
- **Narrow distillation data.** About 1,000 blocks of 2,048 tokens of general text.
- Only about 1.25% of the parameters (338M) are trainable; the rest is frozen by design.

## Gates

The project is judged against fixed milestones, measured automatically at the end of each run:

1. one-step arithmetic: more than half correct
2. GSM8K: more than half correct
3. GPQA Diamond above a dense 4B model (76.2)
4. GPQA Diamond above a dense 9B model (81.7) — the goal. The 27B original scores 89.2.

None has been passed yet.

## Next

- **Budget market (v14, ready, not yet run on the 27B).** The FFNs run at ~7% density and the attention
  projections at ~30%, yet both cost about the same. The 3.0B is re-divided by measurement: nudge each
  family/depth knob, read the held-out KL, move budget to where it buys the most. No gradient steps.
- **Loss-aware lens.** Select the neurons that change the final answer, not the ones with the largest activation.
- **A lens for attention**, the least reworked part of the model.
- A real sparse execution kernel, once the read pattern is settled.

## Reproducing

Each version is a self-contained notebook. v12 and later restore the previous version's saved state instead
of rebuilding from scratch.

- Hardware: one GPU with 80 GB or more (A100 80GB, H100, H200).
- v10: about 7 hours. v12: about 30 minutes. v13: about 3 hours.

## Base model and license

Base model: Qwen3.8-27B by Alibaba Cloud (Apache-2.0). Reference scores for the dense 4B / 9B / 27B models
are taken from their public model cards.

Code in this repository: Apache-2.0.
