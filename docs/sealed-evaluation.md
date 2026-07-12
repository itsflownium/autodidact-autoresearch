# Sealed evaluation and reporting

Sealed evaluation answers a different question from PatchRCT promotion. PatchRCT decides whether a
patch may become the next research parent using public-development and promotion evidence. The
sealed workflow is opened only after research lineages are frozen, and it measures how many
promotions survive a final untouched split.

## Two-phase boundary

### 1. Freeze a plan

`autodidact-sealed plan` reads each arm's verified ledger while requesting only the public dataset
scope. It freezes:

- the common initial Git parent;
- every promoted commit in every accepted lineage;
- the ledger ID and final event-chain hash;
- predetermined evaluation seeds and execution assignment seed;
- full training, batch, timeout, device, and parameter contracts;
- public dataset, protected evaluator, and sealed-runner hashes; and
- campaign compute already spent by each arm.

The plan is canonical JSON with a SHA-256 sidecar. Planning fails if arms do not share one initial
parent or if any accepted lineage is discontinuous. It does not open, inspect, or evaluate the
protected split.

```bash
uv run autodidact-sealed \
  --repository-root . \
  --sealed-root artifacts/sealed/pilot-001 \
  plan \
  --arm greedy=artifacts/studies/pilot-001/arms/greedy/ledger.sqlite3 \
  --arm patch-rct=artifacts/studies/pilot-001/arms/patch_rct/ledger.sqlite3 \
  --arm patch-rct-bayesian=artifacts/studies/pilot-001/arms/patch_rct_bayesian/ledger.sqlite3 \
  --target-config artifacts/control/target.json \
  --seeds 101 211 307 \
  --assignment-seed 20260712
```

Stop all research campaigns after creating this plan. If a ledger, accepted parent, evaluator,
runner, or public data commitment changes, execution fails closed instead of silently updating the
claim.

### 2. Execute the frozen plan

```bash
uv run autodidact-sealed \
  --repository-root . \
  --sealed-root artifacts/sealed/pilot-001 \
  run
```

Execution verifies the complete protected dataset, then trains every unique accepted commit from
scratch on each declared seed. Training receives a read-only public-only dataset view. The
protected evaluator alone receives `sealed_final` after the checkpoints exist.

Identical commit/seed combinations are deterministic and retained once. A promoted commit's sealed
checkpoint is reused as the next generation's parent checkpoint for the same seed. Partial runs
retain their contract and checkpoint so the command can recover without changing the plan.

No real campaign or sealed run is part of the source repository. Checkpoints, process logs, and raw
sealed measurements remain ignored runtime artifacts.

## Reports

`run` writes the report after every frozen result is available. It can also be regenerated without
training:

```bash
uv run autodidact-sealed \
  --sealed-root artifacts/sealed/pilot-001 \
  report
```

The report directory contains:

| Artifact | Purpose |
| --- | --- |
| `report.json` | Complete machine-readable arm, generation, transition, and compute summaries |
| `report.md` | Human-readable study result and promotion confirmation table |
| `promotions.csv` | One row per promoted transition for downstream analysis |
| `sealed-results.svg` | Mean sealed BPB across accepted generations |
| `manifest.json` | SHA-256 and size of every published report artifact |

For each promotion, paired gain is the previous accepted generation's sealed BPB minus the new
generation's sealed BPB on the same seed.

- `useful_confirmed`: the paired Student-t 95% lower bound reaches the patch's predeclared
  minimum useful gain.
- `false_promotion`: mean paired sealed gain is nonpositive.
- `unconfirmed`: the point estimate is positive but evidence does not clear the useful-effect
  threshold.

The report also records final lineage gain, false-promotion rate, campaign compute, and compute per
useful confirmed promotion. These labels summarize frozen evidence; they never modify research
lineage or retroactively change a PatchRCT decision.
