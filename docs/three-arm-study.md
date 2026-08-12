# Three-arm autoresearch study

The study harness compares three decision architectures from one exact Git parent under shared hard
ceilings for proposals, researcher tokens, target-training units, wall time, and compute time.

| Arm | Keep or discard behavior | Evidence path |
| --- | --- | --- |
| `greedy` | Keep after one fixed cheap pair has positive observed gain. | Immediate keep/discard control with the same protected runner. |
| `patch_rct` | Promote only when direct full-stage evidence clears useful-effect and resource gates. | Cheap, intermediate, and full paired stages. |
| `patch_rct_bayesian` | Use PatchRCT for promotion and a calibrated learning-curve model for test allocation. | Forced labels, prediction-guided allocation, and deterministic audits. |

The greedy arm is the Karpathy-style control treatment. It still uses immutable target contracts,
protected evaluation, matched parent/candidate worktrees, predetermined seeds, resource checks, and
the evidence ledger. The intended difference is the decision rule, not access to data or compute.

## Isolation

Each arm receives its own ledger, campaign-state database, researcher transcripts, candidate
workspaces, run artifacts, reward-model artifacts, and protected accepted Git ref. Every accepted ref
starts at the same commit. Promotion in one arm cannot change another arm's parent.

The canonical study manifest pins the parent, deterministic arm order, budgets, researcher config,
target config, program, calibration target, and controller policy hashes. Runtime rejects drift.

## Initialize

```bash
uv run autodidact-study \
  --repository-root . \
  --study-root artifacts/studies/pilot-001 \
  initialize \
  --study-id pilot-001 \
  --assignment-seed 20260712 \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  --max-proposals 50 \
  --max-wall-seconds 604800 \
  --max-researcher-tokens 50000000 \
  --max-training-tokens 7400000000 \
  --max-compute-seconds 604800 \
  --reward-calibration-labels 40
```

Initialization creates control state only. It does not invoke a researcher or run a target. The
training-limit option keeps its historical name, but a plugin receives target units through
`{training_budget}` and declares their meaning in `rl.budget_unit` when applicable.

## Run and recover

```bash
uv run autodidact-study \
  --repository-root . \
  --study-root artifacts/studies/pilot-001 \
  run --max-new-proposals-per-arm 1

uv run autodidact-study \
  --repository-root . \
  --study-root artifacts/studies/pilot-001 \
  status
```

The orchestrator durably records each budget reservation, inference transcript, candidate commit,
paired schedule, run, prediction, decision, and lineage before advancing. Repeating `run` resumes
from those records rather than paying for a completed operation again.

Study runtime artifacts remain local and ignored. Publish only the sealed report and an explicitly
redacted evidence export.
