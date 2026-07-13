# Three-arm autoresearch study

The study harness compares three autonomous research systems from the same initial Git parent under
identical hard ceilings for proposals, researcher tokens, training tokens, wall time, and compute
seconds.

| Arm | Keep or discard behavior | Evidence path |
| --- | --- | --- |
| `greedy` | Keep a patch when one fixed cheap paired comparison has positive observed BPB gain. | Karpathy-style immediate keep/discard, using the protected runner and ledger. |
| `patch_rct` | Promote only after direct full-budget PatchRCT evidence passes its probability and resource gates. | Cheap, intermediate, and full paired stages with predetermined seeds. |
| `patch_rct_bayesian` | Use PatchRCT for promotion, but use a calibrated learning-curve model to allocate full tests. | Forced labels first, then protected Bayesian allocation with deterministic audits. |

The greedy arm intentionally does not use PatchRCT's minimum-useful-effect probability to decide
whether to keep a patch. That difference is the control treatment being measured. It still uses
the same immutable dataset, protected evaluator, parent/candidate worktrees, parameter cap,
resource checks, and evidence ledger, so failures and costs remain comparable.

## Isolation

Each arm receives its own:

- append-only experiment ledger;
- durable campaign state database;
- researcher transcripts and candidate workspaces;
- experiment and reward artifacts; and
- accepted Git ref under `refs/autodidact/studies/STUDY_ID/ARM/accepted`.

All three accepted refs initially point to the same exact commit. A promotion in one arm cannot
change another arm's parent. The arm execution order is fixed by hashing the study ID, assignment
seed, and arm name before outcomes exist.

The study manifest is canonical JSON with a SHA-256 sidecar. It commits the initial parent, arm
order, hard budgets, researcher/target configuration paths, calibration target, and the complete
policy hash for every arm. It also pins SHA-256 hashes for the researcher configuration, target
configuration, and research program. Runtime refuses to continue if any input or policy differs
from that manifest.

## Initialize

Prepare the researcher and target configurations first, then create a study. The 50-proposal pilot
reserves 40 Bayesian-arm proposals for full-budget reward labels before prediction-guided
allocation begins.

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
  --max-training-tokens 3000000000 \
  --max-compute-seconds 604800 \
  --reward-calibration-labels 40
```

Initialization creates control state only. It does not invoke a researcher or train a model.

## Run and recover

Advance each arm by at most one new proposal in its preassigned order:

```bash
uv run autodidact-study \
  --repository-root . \
  --study-root artifacts/studies/pilot-001 \
  run --max-new-proposals-per-arm 1
```

The underlying orchestrator retains each operation, budget reservation, transcript, schedule, run,
prediction, decision, and lineage before moving forward. Repeating the command resumes from those
records instead of paying for a completed operation again.

Inspect all arms without running another proposal:

```bash
uv run autodidact-study \
  --repository-root . \
  --study-root artifacts/studies/pilot-001 \
  status
```

Study artifacts are ignored local runtime evidence. Compact reports and an explicitly redacted
ledger export can be published later by the sealed reporting workflow.
