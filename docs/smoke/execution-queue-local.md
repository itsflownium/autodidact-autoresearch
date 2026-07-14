# Execution Queue Local Smoke

This bounded local diagnostic confirms that the fixed proposal queue can drive the real Autodidact orchestration path. It does not establish that proposal 18 improves model quality.

## Scope

- Queue: `autodidact-tinystories-proposals-v1`, rank 1 of 60.
- Assigned source proposal: 18, `Mask cross-story boundary targets`.
- Source parent: `e150af023855c5bc3b0e9c8745c1126e8509258a`.
- Actual parent: `3350006212991f6b135e848f5f1cf23b5dc06be9`.
- Device: local MPS.
- Model: 1,016,960 trainable parameters.
- Decision policy: greedy diagnostic, not PatchRCT.
- Pair: seed 11, 8,192 training tokens and 8,192 requested evaluation tokens per arm.

The deterministic smoke researcher received the protected queue assignment, recognized that the parent had changed, applied the same stored mechanism to the current `train.py`, and returned the frozen proposal claim. The orchestrator committed and registered that candidate before launching the real paired runner.

## Result

The parent and candidate each completed 8,192 training tokens and 8,095 document-aligned evaluation tokens. The parent measured 2.8389281435 BPB and the candidate measured 2.8387700482 BPB, an observed one-seed gain of 0.0001580953 BPB. Greedy mode therefore recorded a diagnostic promotion. This tiny result is dominated by seed and short-run noise and is not evidence of a useful or repeatable improvement.

The pair consumed 44.81 accelerator-seconds. The protected ledger contains 14 hash-chained events covering the proposal, candidate, schedule, trial, two runs, two artifact manifests, two compute records, paired result, effect estimate, decision, and lineage. Ledger verification passed at head `0b9654681a62d9ea2f80f2306f6e58a1fce15e889c69c7f54c0af54018dbf736`.

All 12 retained artifacts, totaling 24,520,991 recorded bytes, were found and matched their recorded size and SHA-256. The dataset staging tree uses hard links to the prepared cache, so its apparent 1.1 GiB size is not an additional dataset copy.

## Recovery

The completed campaign and ledger were reopened in a new process and `run` was called again. It returned zero outcomes, performed no new research or training, retained the same 16,384 training-token and 44.81-second usage totals, and left the ledger head unchanged.

The machine-readable record is [`execution-queue-local.json`](execution-queue-local.json).

## Claim Limit

This diagnostic establishes queue loading, patch adaptation, orchestration, protected paired execution, ledger recording, artifact saving, restart, and idempotent replay. A quality claim requires the predeclared PatchRCT stages, repeated paired seeds, larger budgets, uncertainty thresholds, resource gates, and full-budget confirmation.
