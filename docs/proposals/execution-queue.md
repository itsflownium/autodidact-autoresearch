# Proposal Execution Queue

This queue covers all 60 fixed-parent proposals. Every item remains **queued and unmeasured**.
Rank controls compute triage only; it is not evidence of quality. After any promotion, the
configured researcher must revalidate or adapt the next patch against the accepted parent.

- Queue ID: `autodidact-tinystories-proposals-v1`
- Frozen proposal parent: `e150af023855c5bc3b0e9c8745c1126e8509258a`
- Source banks: `bank-001-030.json`, `bank-031-060.json`
- Tiers: ranks 1-20 `screen`, 21-45 `explore`, 46-60 `defer`

## Ranking policy

- Queue rank is compute triage, not evidence that a patch is better.
- Screen low-disruption and high-information hypotheses before parameter-heavy structural variants.
- Place one representative from major conflict families early and defer redundant alternatives.
- After every promotion, revalidate or adapt all remaining patches against the new accepted parent.
- Never use queue order to select seeds, override protected outcomes, or claim model quality.
- Tie breaker: Prefer lower adaptation risk, then lower resource risk, then lower original proposal number; no measured outcome was used.

## Queue

| Rank | Proposal | Tier | Adapt | Resource | Title | Conflict groups |
| ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 018 | screen | low | low | Mask cross-story boundary targets | `data-locality`, `auxiliary-objective` |
| 2 | 007 | screen | low | low | No weight decay on tied embeddings | `optimizer-schedule`, `embedding-parameterization` |
| 3 | 027 | screen | low | low | Reduce AdamW weight decay to 0.01 | `optimizer-schedule` |
| 4 | 038 | screen | low | low | Increase AdamW beta2 to 0.98 | `optimizer-schedule` |
| 5 | 017 | screen | low | low | Shorten learning-rate warmup to 1% | `optimizer-schedule`, `learning-rate-curve` |
| 6 | 014 | screen | low | low | Cosine learning-rate decay to zero | `optimizer-schedule`, `learning-rate-curve` |
| 7 | 019 | screen | low | low | Double the peak learning rate | `optimizer-schedule`, `learning-rate-curve` |
| 8 | 022 | screen | low | low | Final-20% cosine decay | `optimizer-schedule`, `learning-rate-curve` |
| 9 | 011 | screen | low | low | Halve the training batch to 32 | `optimizer-schedule` |
| 10 | 031 | screen | low | low | Raise gradient clipping threshold to 2.0 | `optimizer-schedule` |
| 11 | 006 | screen | medium | medium | Near-parameter-matched SwiGLU MLP | `activation-choice`, `mlp-structure` |
| 12 | 004 | screen | medium | medium | Parameter-free query-key RMS normalization | `attention-topology` |
| 13 | 005 | screen | medium | medium | Residual-stream RMSNorm | `core-normalization` |
| 14 | 002 | screen | low | low | Short-context RoPE base 1,000 | `rope-variants` |
| 15 | 046 | screen | medium | medium | Hybrid local-global attention heads | `attention-topology`, `data-locality` |
| 16 | 050 | screen | low | medium | Token-embedding dropout | `embedding-parameterization` |
| 17 | 058 | screen | low | low | Width-scaled token embeddings | `residual-layout-init`, `embedding-parameterization` |
| 18 | 033 | screen | high | high | Reset attention at story boundaries | `attention-topology`, `data-locality` |
| 19 | 044 | screen | medium | medium | Early context-length curriculum | `optimizer-schedule`, `data-locality` |
| 20 | 053 | screen | high | high | Strided midpoint deep supervision | `output-objective`, `auxiliary-objective` |
| 21 | 024 | explore | high | high | Parameter-neutral multi-query attention reallocation | `attention-topology`, `mlp-structure` |
| 22 | 041 | explore | high | high | Wider 136-channel residual stream | `mlp-structure`, `parameter-budget` |
| 23 | 045 | explore | high | high | Factorized embeddings with MLP reallocation | `mlp-structure`, `embedding-parameterization` |
| 24 | 016 | explore | high | high | Five-layer depth-for-width transformer | `mlp-structure`, `block-layout`, `parameter-budget` |
| 25 | 054 | explore | high | high | Sandwich half-width MLP sublayers | `mlp-structure`, `residual-layout-init`, `block-layout` |
| 26 | 009 | explore | high | high | Widen MLPs to the parameter cap | `mlp-structure`, `parameter-budget` |
| 27 | 003 | explore | high | high | Learned absolute position embeddings | `position-replacement`, `embedding-parameterization`, `parameter-budget` |
| 28 | 039 | explore | high | medium | Replace RoPE with fixed ALiBi | `position-replacement`, `attention-topology` |
| 29 | 060 | explore | medium | medium | Hybrid RoPE and position-free attention heads | `rope-variants`, `attention-topology` |
| 30 | 013 | explore | low | low | Half-dimensional rotary embeddings | `rope-variants` |
| 31 | 026 | explore | medium | medium | Per-head learnable RoPE frequencies | `rope-variants`, `parameter-budget` |
| 32 | 035 | explore | medium | medium | Bucketed learned relative attention bias | `rope-variants`, `attention-topology`, `parameter-budget` |
| 33 | 012 | explore | low | low | Two wider 64-dimensional attention heads | `attention-topology` |
| 34 | 023 | explore | medium | medium | Learnable per-head attention temperatures | `attention-topology`, `parameter-budget` |
| 35 | 051 | explore | low | low | Order-one query-key logit initialization | `attention-topology` |
| 36 | 057 | explore | medium | medium | Parameter-free query-gated attention | `attention-topology` |
| 37 | 021 | explore | medium | medium | Causal depthwise convolution in each MLP | `mlp-structure`, `data-locality`, `parameter-budget` |
| 38 | 029 | explore | medium | medium | Causal depthwise convolution on attention values | `attention-topology`, `data-locality`, `parameter-budget` |
| 39 | 047 | explore | medium | medium | Parameter-free causal MLP channel shift | `mlp-structure`, `data-locality` |
| 40 | 040 | explore | medium | medium | Learned causal previous-token embedding mix | `data-locality`, `embedding-parameterization`, `parameter-budget` |
| 41 | 032 | explore | medium | medium | Learned causal prefix sink | `attention-topology`, `parameter-budget` |
| 42 | 042 | explore | medium | medium | Parallel attention and MLP residual branches | `residual-layout-init`, `block-layout` |
| 43 | 025 | explore | medium | medium | Per-channel residual branch scales | `residual-layout-init`, `parameter-budget` |
| 44 | 049 | explore | medium | medium | Per-layer token-embedding reinjection | `residual-layout-init`, `embedding-parameterization`, `parameter-budget` |
| 45 | 055 | explore | medium | medium | Per-channel intermediate-depth readout mixing | `residual-layout-init`, `parameter-budget` |
| 46 | 001 | defer | medium | low | Squared-ReLU transformer MLP | `activation-choice`, `mlp-structure` |
| 47 | 059 | defer | low | low | SiLU transformer MLP activation | `activation-choice`, `mlp-structure` |
| 48 | 034 | defer | medium | medium | Post-activation MLP hidden normalization | `mlp-structure`, `parameter-budget` |
| 49 | 037 | defer | low | low | Lower LayerNorm epsilon to 1e-6 | `core-normalization` |
| 50 | 030 | defer | medium | medium | Fully affine LayerNorm | `core-normalization`, `residual-layout-init`, `parameter-budget` |
| 51 | 010 | defer | medium | medium | Zero-initialized transformer projection biases | `residual-layout-init`, `residual-output-init`, `parameter-budget` |
| 52 | 043 | defer | low | low | Unscaled residual projection initialization | `residual-layout-init`, `residual-output-init` |
| 53 | 056 | defer | low | low | Zero-initialized residual output projections | `residual-layout-init`, `residual-output-init` |
| 54 | 008 | defer | medium | medium | Learned vocabulary-logit bias | `output-objective`, `embedding-parameterization`, `parameter-budget` |
| 55 | 028 | defer | medium | medium | Learned global logit temperature | `output-objective`, `parameter-budget` |
| 56 | 052 | defer | medium | medium | Per-token tied-output logit scales | `output-objective`, `embedding-parameterization`, `parameter-budget` |
| 57 | 020 | defer | high | high | Low-rank untied output correction | `output-objective`, `embedding-parameterization`, `parameter-budget` |
| 58 | 036 | defer | high | high | Low-rank bigram logit shortcut | `output-objective`, `embedding-parameterization`, `parameter-budget` |
| 59 | 015 | defer | medium | medium | Training-time logit z-loss | `output-objective`, `auxiliary-objective` |
| 60 | 048 | defer | high | high | Auxiliary two-token prediction head | `output-objective`, `auxiliary-objective`, `parameter-budget` |

## Conflict groups

### `activation-choice`: Transformer MLP activation choice

- Severity: `exclusive`
- Members: 001, 006, 059
- Rationale: These patches replace the same baseline GELU path with incompatible activation or gated-MLP choices; after one promotion, the others must be reformulated as direct alternatives.

### `position-replacement`: Positional representation replacement

- Severity: `exclusive`
- Members: 003, 039
- Rationale: Learned absolute positions and fixed ALiBi each remove baseline RoPE in different ways and cannot be applied together as written.

### `rope-variants`: RoPE geometry variants

- Severity: `rebase_required`
- Members: 002, 013, 026, 035, 060
- Rationale: These proposals alter overlapping RoPE or relative-position code. A promoted member changes the baseline for every later member even when a combination remains conceptually possible.

### `core-normalization`: Core normalization formulation

- Severity: `exclusive`
- Members: 005, 030, 037
- Rationale: RMSNorm, fully affine LayerNorm, and a LayerNorm epsilon change target the same core normalization sites; later alternatives may become obsolete or require a new claim.

### `optimizer-schedule`: Optimizer and training schedule

- Severity: `interaction`
- Members: 007, 011, 014, 017, 019, 022, 027, 031, 038, 044
- Rationale: Batch size, optimizer moments, clipping, regularization, warmup, learning rate, decay, and context curriculum jointly determine optimization; each result is conditional on earlier accepted recipe changes.

### `learning-rate-curve`: Learning-rate curve

- Severity: `rebase_required`
- Members: 014, 017, 019, 022
- Rationale: These patches edit overlapping learning-rate defaults or schedule logic and must be adapted to any previously accepted curve change.

### `output-objective`: Output parameterization and training objective

- Severity: `interaction`
- Members: 008, 015, 020, 028, 036, 048, 052, 053
- Rationale: Logit biases, scales, residual heads, and auxiliary losses alter the same output geometry or gradients, so later effect estimates depend on earlier accepted members.

### `attention-topology`: Attention topology and logits

- Severity: `rebase_required`
- Members: 004, 012, 023, 024, 029, 032, 033, 035, 039, 046, 051, 057, 060
- Rationale: These patches share attention projections, masks, head layout, initialization, or positional logits. Accepted changes require code adaptation and a fresh interaction assessment.

### `mlp-structure`: MLP structure and capacity

- Severity: `rebase_required`
- Members: 001, 006, 009, 016, 021, 024, 034, 041, 045, 047, 054, 059
- Rationale: These proposals alter MLP activation, width, placement, local mixing, or the architecture budget assigned to MLPs; they cannot be replayed blindly after a promotion.

### `residual-layout-init`: Residual layout and initialization

- Severity: `rebase_required`
- Members: 010, 025, 030, 042, 043, 049, 054, 055, 056, 058
- Rationale: These proposals modify residual branches, normalization, scale, initialization, or readout mixing and therefore share both code surfaces and optimization assumptions.

### `residual-output-init`: Residual output initialization

- Severity: `exclusive`
- Members: 010, 043, 056
- Rationale: Projection biases, unscaled random initialization, and zero residual initialization are competing initialization formulations for the same output projections.

### `block-layout`: Transformer block layout

- Severity: `exclusive`
- Members: 016, 042, 054
- Rationale: Depth-for-width, parallel branches, and sandwich MLPs change the block graph in ways that invalidate the other patches as direct fixed-parent alternatives.

### `data-locality`: Document boundaries and local token mixing

- Severity: `interaction`
- Members: 018, 021, 029, 033, 040, 044, 046, 047
- Rationale: These proposals change which local or cross-story token relationships are modeled, so their gains may overlap and must be re-estimated after a related promotion.

### `auxiliary-objective`: Auxiliary and masked training objectives

- Severity: `interaction`
- Members: 015, 018, 048, 053
- Rationale: Target masking, z-loss, two-token prediction, and midpoint supervision alter training gradients and must use mutually consistent ignored-token semantics.

### `embedding-parameterization`: Embedding and tied-output parameterization

- Severity: `rebase_required`
- Members: 003, 007, 008, 020, 036, 040, 045, 049, 050, 052, 058
- Rationale: These patches alter embedding scale, regularization, dimensionality, tied-output behavior, or embedding reinjection and require adaptation after any accepted member.

### `parameter-budget`: One-million-parameter budget contention

- Severity: `budget_recheck`
- Members: 003, 008, 009, 010, 016, 020, 021, 023, 025, 026, 028, 029, 030, 032, 034, 035, 036, 040, 041, 048, 049, 052, 055
- Rationale: Each member consumes baseline parameter headroom. After one promotion, every later member must be re-budgeted against the 1,050,000-parameter cap.

## Execution rule

A queued patch may be applied exactly only when it still fits the current accepted parent.
Otherwise the configured researcher must preserve the same atomic hypothesis while adapting
the implementation. If that cannot be done without bundling another change, the item returns
`no_change`; it is never silently replaced by a different idea.
