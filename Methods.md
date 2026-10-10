# Methods

Notation: `x` a token's hidden state, `d = 5120`, FFN width `D = 17408`, vocabulary `V = 248,320`. The dense teacher is
the same frozen weights run without any selection ("dense world"); the student is those weights read through the
mechanisms below ("served world").

## The counting rule
A census walks the served forward pass and counts every parameter a single token reads: exact rows and columns,
routers, predictors, low-rank shadows, the output head and its shortlist, norms. The limit is **≤ 3.0B for every
token**. Since v16 the census is checked against what the forward pass actually executes (v16 found two places where it
was not: a dense output head and averaged memory writes).

## Neuron lens (FFN)
A rank-128 predictor `â(x)` estimates all `D` activations. A token's exact neurons are the top-scoring ones under
`score_i = |â_i − μ_i| · ||W_down[:, i]|| · e^{w_i(x)}` (`w` = the trained "consequence head"). Exact neurons use the
original gate/up/down weights; an unselected neuron contributes its mean `μ_i`; a closed-form low-rank shadow models the
rest. "Own neurons" (v16) are trainable copies of the 256 most-used neurons per layer.

## Shadow execution (attention projections)
`y = S(x) + Σ_selected (exact_rows(x) − S_rows(x))`: a low-rank estimate everywhere, exact row-blocks where a router
selects them.

## Strict memory writes (Gated-DeltaNet layers)
Exactly `m = 4` of 48 value heads write exactly per token; the others write their shadow. A critic predicts each head's
influence `L = e^{s}`; the top-m are chosen. The straight-through gradient is taken in log-score space,
`σ((s − s_(m)) / τ)`, so its size is bounded by `1/(4τ)` (in score space it grew with `e^{s}` and overflowed, v16.1).

## Strict head + LIFT (output layer)
Proxy logits `z̃ = (xP)Uᵀ + b` (rank 256) rank the whole vocabulary; the top `k ≈ 6,000` rows are computed exactly,
everything else is served by the proxy. The loss runs through this served head chunk by chunk. LIFT trains only the
proxy: for teacher top-K tokens `j` with probabilities `p_j`,
`L_LIFT = Σ_j p_j · softplus(z̃_(k) + margin − z̃_j)`, i.e. the teacher's tokens are lifted above the shortlist line.

## The wallet (per-token budgets)
Every selectable site (FFN neurons per layer, row-blocks of routed projections, memory writes) has a quota `n₀` and a
price threshold `θ`. Each token starts with its own balance `b₀ = ceiling − census at the quotas`. At a site, a token
buys `n = clamp(#{u ≥ θ − pace(balance)}, lo, min(hi, n₀ + ⌊(balance + ρ·refund)/cost⌋))` items and pays
`cost · (n − n₀)`; its balance can never end below zero, so every token stays under the ceiling. Prices move by
tâtonnement: a zero-valued probe per site measures `−∂L/∂probe / (cost · mass)`, the marginal value of one more
parameter there, and sites worth more than the median lower their price. Since v17 the overall price level follows the
budget tokens leave unspent. Since v18 a token may sell at most one memory write and one row-block per attention
projection.

## In-step teacher and full KL (v17)
The dense teacher runs on every training batch (it already did, for the stream anchor). Its final hidden state gives
the full distribution `p_t`, and the loss is `KL(p_t || q_s)` over all `V` tokens, computed chunk by chunk inside the
served head with gradient `q_s − p_t` (checked against autograd, relative error 1.5e-7). No teacher cache is built,
so the data size is no longer limited by caching.

## DRIFT (v17)
Teacher forcing never shows a student its own errors. In half the micro-batches the student first reads the batch
without gradient; where its argmax differs from the next token of the text, that token is replaced by the student's
choice (half of those positions, ≤ 6% of a sequence). The wrong token is input only, never a target: at its own
position the teacher's distribution still points to the right token, and after it the teacher shows how to continue.
(Imitation-learning theory: errors compound as `O(T²ε)` under teacher forcing and `O(Tε)` when the expert labels the
learner's own states.) From v18, problem rows and LOOP rows are not drifted.

## LOOP (v17)
Once every three updates the student answers 24 fresh arithmetic problems with greedy, batched, cached decoding in the
served world. The exchanges (prompt, the student's tokens, end of turn) form a training row; the teacher labels every
position, so the student is corrected exactly at the digits it gets wrong.

## Fork loss (v16)
`L_fork = −(1 − q) · log q`, `q` = the student's probability on the teacher's acceptable tokens (≥ α × the teacher's
best). A position the student already gets right contributes nothing.

## Fuse (v16.2)
After every backward, a loss or gradient that is not finite never reaches the weights: a few affected tensors sit the
update out, many → the update is skipped; the deepest affected tensor is printed (that is where it starts).
