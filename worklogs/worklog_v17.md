# Work log — v17 (2026-10-09 / 10)

## Plan
Five changes on top of v16.2, restored from the v16.2 state:
1. **In-step teacher.** The dense teacher already ran in every training step for the stream anchor. The same pass now
   labels the batch, so no teacher cache is built and the data can grow.
2. **Full KL.** The loss is the KL to the teacher's whole 248k-token distribution, computed chunk by chunk inside the
   served head (gradient checked against autograd: relative error 1.5e-7). The top-64 KL is still logged.
3. **DRIFT.** In every other micro-batch the student first reads the batch (served, no gradient). Where its first
   choice differs from the text, the next token is replaced by the student's choice (half of those positions, at most
   6% of a sequence). The teacher then labels the batch as it now reads.
4. **LOOP.** Once every three updates the student writes answers to 24 fresh arithmetic problems (batched, cached
   decoding); the teacher grades every token it wrote. The math probe's own problems are excluded.
5. **Wallet fixes.** The price level follows the unspent budget (v16.2 left ~157M per token unspent). Ranges widened:
   FFN 0–2× the quota (was 0.5–1.5×), +6 row-blocks (was +3), +8 memory writes (was +4).

Also: stream anchor 1.0 → 0.25; 1,022 blocks of fresh text from sources no version had read; KL by source in the log;
a long-context curve (KL by position up to 32K tokens).

## Run (one A100 80GB, 236 steps, 202 minutes, stopped by the wall-clock budget)

| Step | score | text KL | reasoning KL | decisive matched | 12 in a row |
|---|---|---|---|---|---|
| 0 (v16.2 state) | 2.130 | 1.862 | 0.540 | 91.4% | 46.2% |
| 50 | 2.137 | 1.801 | 0.671 | 85.2% | 24.7% |
| 100 | 2.068 | 1.769 | 0.598 | 89.2% | 32.9% |
| 150 | 2.006 | 1.717 | 0.578 | 89.7% | 32.9% |
| 200 | **1.962** | **1.699** | **0.528** | **91.9%** | 42.4% |
| 236 | 1.963 | 1.700 | 0.527 | 90.8% | 35.9% |

(small eval set; score = text KL + 0.5 × reasoning KL; "reasoning" here is held-out problems only — see below)

**Final.** Held-out PPL 33.2 → **28.94** (v15, with the uncounted dense head: 31.35). Head shortlist holds 98.0% of the
teacher-head probability; the dense head's first choice is inside it 100% of the time. One-step arithmetic 3/20.

**LOOP** — the student's own answers, correct per 25 updates (n ≈ 192 each):
13, 8, 8, 13, 20, 8, 23, 21, 25 → 7% at the start, 13% at the end.

**Wallet.** Mean parameters per token 2.84B (v16.2) → 2.98B. Unspent budget 38.9M → 3–7M per token. Per-token range on a
held-out block 2.84B – 2.9996B. FFN neurons bought: mean 555 (quota 482), from 286 to 723 per token.

**Memory.** Peak 71.2 GiB of 79 at every step; zero fuse events.

## Findings
- **The reasoning traces never reached training** (this run nor v16.2). The loader fell back to reading the whole
  dataset repository, met a pre-tokenized folder with another schema (CastError), and also looked for a `messages`
  column where the set has `messages_json`. The data cell printed one line and went on. The "reasoning" numbers of v16.2
  and v17 are therefore problems (arithmetic + GSM8K-train style) only. Fixed in v18.
- **DRIFT hurt arithmetic early.** At step 50 "12 decisive positions in a row" had fallen 46% → 25%; it came back to 42%
  by step 200. Rewriting digits in problem rows turns them into inconsistent calculations. v18 keeps DRIFT out of
  problem and LOOP rows.
- **The wallet sold the long-range machinery.** On a held-out block a token kept 1.0 of 4 exact memory writes,
  0.4 of 1.8 query row-blocks and 0.4 of 1.2 z row-blocks, and bought FFN neurons. The x-ray agrees: FFN error
  0.728 → 0.703, but linear-attention out_proj 0.331 → 0.421 and every other attention part rose. v18 sets floors.
- **Wallet ceiling.** The largest per-token reading, 2.9996B, is 0.6M above the wallet's internal ceiling (2.9990B)
  and 0.4M below the 3.0B law. The accounting gap appeared with the wider ranges; not yet explained.
- **No long-context curve.** The long document was to come from the trace set, which never loaded.

## Next (v18)
Reasoning traces loaded for real and trained in 2,048-token rows; memory writes and attention row-blocks protected
from being sold; GSM8K (20), arithmetic (100) and MMLU-Pro (140, 14 subjects) measured every run; the long-context
curve on a real long document.
