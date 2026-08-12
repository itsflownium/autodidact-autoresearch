# Sealed evaluation and reporting

PatchRCT promotion and sealed evaluation answer different questions. PatchRCT decides which patch
becomes the next research parent using development and promotion evidence. Sealed evaluation runs
only after lineages are frozen and measures whether those promotions survive untouched tasks or
data.

The same workflow supports supervised metrics, environment rewards, and verifier rewards because
it uses the configured target plugin. Autodidact does not provide a sealed dataset or verifier.

## 1. Freeze a plan

```bash
uv run autodidact-sealed \
  --repository-root . \
  --sealed-root artifacts/sealed/pilot-001 \
  plan \
  --arm greedy=artifacts/studies/pilot-001/arms/greedy/ledger.sqlite3 \
  --arm patch-rct=artifacts/studies/pilot-001/arms/patch_rct/ledger.sqlite3 \
  --arm patch-rct-bayesian=artifacts/studies/pilot-001/arms/patch_rct_bayesian/ledger.sqlite3 \
  --target-config artifacts/control/target.json \
  --token-budget 20000000 \
  --seeds 101 211 307 \
  --assignment-seed 20260712
```

Planning reads verified public ledgers and freezes:

- the common initial Git parent and every promoted commit;
- each ledger identity and final event-chain hash;
- predetermined seeds and execution assignment;
- training budget, batch, timeout, device, and parameter contracts;
- plugin, evaluator, public-source, and protected-source commitments; and
- campaign compute already spent by each arm.

The plan is canonical JSON with a SHA-256 sidecar. It does not run the target or open protected
inputs. Stop research after freezing it. Later drift in a ledger, accepted lineage, target contract,
or evaluator fails closed.

## 2. Execute

```bash
uv run autodidact-sealed \
  --repository-root . \
  --sealed-root artifacts/sealed/pilot-001 \
  run
```

Execution checks out each unique accepted commit in a clean detached worktree, trains it under each
declared seed, and calls the protected evaluator with the `sealed_final` split. RL and RLVR runs must
also satisfy the reward range, reward-source, reward-variance, and verifier-coverage contract.

A completed commit/seed result is retained once and reused when multiple arms share it. Repeating
the command verifies retained contracts and resumes missing runs instead of duplicating completed
work. Source control contains no checkpoints or raw sealed outcomes.

## Reports

```bash
uv run autodidact-sealed \
  --sealed-root artifacts/sealed/pilot-001 \
  report
```

| Artifact | Purpose |
| --- | --- |
| `report.json` | Machine-readable arm, generation, transition, and compute summaries |
| `report.md` | Human-readable results and promotion confirmation table |
| `promotions.csv` | One row per promoted transition |
| `sealed-results.svg` | Mean canonical objective across accepted generations |
| `manifest.json` | Hash and size of every report artifact |

For each promotion, paired gain is the previous accepted generation's sealed canonical objective
minus the new generation's objective on the same seed.

- `useful_confirmed`: the paired 95% lower confidence bound reaches the predeclared minimum useful
  gain.
- `false_promotion`: the mean paired sealed gain is nonpositive.
- `unconfirmed`: the point estimate is positive but uncertainty still crosses the useful threshold.

Reports also include final lineage gain, false-promotion rate, campaign compute, and compute per
confirmed promotion. Sealed labels do not rewrite the historical lineage or retroactively change a
PatchRCT decision.
