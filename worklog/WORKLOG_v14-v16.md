# Work log — v14 to v16.2

## v14 — training-free budget market
- Moved the 3.0B between ~30 coarse knobs (family × depth, ±25%) by nudging each knob and reading the held-out KL.
  No gradient steps.
- KL to teacher 2.17 → 2.15. Small gain: the knobs were too coarse to matter much.

## v15 — consequence head, attention shadows
- 322 training steps on one A100 80GB (about 22 Colab compute units).
- Held-out PPL 40.65 → 31.35. Local FFN error 0.708 → 0.631.
- First non-zero arithmetic result: 1/20 (was 0/20 for every earlier version).
- Later found (v16 audit): these numbers used the dense output head. See the README correction.

## v16 — the law (2026-10-07)
**Audit.** Rebuilt the measurement so the forward pass and the census agree.
- Found the output head running dense (1.27B uncounted) and memory writes counted as an average (tokens wrote
  0–36 of 48 heads). Both closed. Strict writes cost ≈ 0 KL; the strict head costs +1.1 text / +2.0 reasoning KL.
- Only 24% of the 3B is selectable per token; 76% is fixed overhead (FFN 59%, linear attention 85%, full
  attention 91%, head 76% of their parts).
- **Dense swap** (one family dense, everything else sparse): every family made the model *worse* (FFN: +1.09
  text KL). The student had co-adapted; it is no longer "teacher plus patches".

**New mechanisms.** Strict head with a trainable proxy; strict memory writes; own neurons (trainable copies of
the 256 most-used FFN neurons per layer); fork loss; per-token wallet with gradient-probe prices.

**v16.1 run (failed).**
- Crash at scheduler creation: the optimizer wrapper replaced `optimizer.step` with a plain function, but
  `LambdaLR` needs a bound method. Fixed with `types.MethodType`; the test scheduler now checks this too.
- NaN at step 2. Root cause found in v16.2 (below).

## v16.2 — head first (2026-10-09)
**Changes.**
- Head rows 4,096 → 5,888, using the free budget under 3.0B (+9M parameters). Before any training the
  starting score improved 4.25 → 3.54.
- **LIFT**: the head proxy is trained only to lift the teacher's tokens above the shortlist line.
- Higher learning rates for the new parts; fork loss weight ×2.
- **Fuse**: a non-finite loss or gradient never reaches the weights; the deepest affected tensor is named.

**NaN root cause.** The memory-write mask's straight-through gradient was computed in score space, so its size
grew with e^score (measured 1.3×10¹⁶ on a synthetic test). Two real runs blew up at layers 45 and 29. Moved to
log-score space: gradient bounded, same choice of heads. After the fix: zero fuse events in 310 steps.

**Run.** 310 steps, 201 minutes, one A100 80GB, stopped by the wall-clock budget. Memory peak 71.2 GiB of 79.

| Step | score | text KL | reasoning KL | runs of 12 |
|---|---|---|---|---|
| 0 | 3.543 | 2.486 | 2.112 | 21.3% |
| 50 | 2.375 | 2.049 | 0.653 | 31.4% |
| 100 | 2.269 | 1.961 | 0.618 | 35.5% |
| 200 | 2.197 | 1.908 | 0.579 | 45.0% |
| 310 | 2.132 | 1.862 | 0.540 | 46.2% |

(small eval set; score = text KL + 0.5 × reasoning KL)

**Final (full held-out set).** PPL 143.32 → 33.40; text KL 3.26 → 1.96; reasoning KL 1.90 → 0.58; decisive
positions 83.7% → 91.1%; head shortlist mass 85.7% → 97.2%; arithmetic 2/20.

**Observations.**
- Local layer errors went *up* (FFN 0.631 → 0.730) while the output got much closer to the teacher: the parts
  learned to work together rather than copy their dense originals.
- The wallet moved budget on its own: tokens sold attention-side rows and memory writes and bought FFN neurons
  (per-token range 268–723 neurons per layer).
- Wallet price rule leaves ~157M per token unspent (prices react to wants of already-empty wallets).

**Next (v17).** Learning from its own answers on arithmetic; wallet price rule fix; fresh data each step with
teacher targets taken in-step; long-context curve; per-source KL.
