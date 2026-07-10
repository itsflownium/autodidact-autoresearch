# Autodidact Research Program

## Mission

Improve the current 1,016,960-parameter TinyStories training implementation through small, attributable code changes. The objective is lower held-out bits per byte under fixed experiment budgets without exceeding runtime, memory, stability, integrity, or parameter constraints.

You propose research patches. The protected experiment controller measures them and decides whether to reject them, gather more evidence, or promote them. You do not grade or promote your own work.

## Authority Boundary

You may edit exactly one repository file:

```text
train.py
```

Do not modify, create replacements for, or bypass any other repository file. In particular, do not change:

- `prepare.py`
- `program.md`
- `pyproject.toml`
- `uv.lock`
- `.github/`
- `autodidact/`
- `tests/`
- dataset manifests, tokenizer artifacts, shards, indexes, or policies

Do not add dependencies, install alternate packages, change the environment, invoke network services, or write outside the experiment workspace. Do not inspect or infer evaluator-only promotion data, sealed-final data, hidden seeds, or hidden decisions. The training process receives only public `train` and `dev` data.

The controller verifies file scope, protected hashes, the command-line interface, data commitments, and the parameter cap before accepting any measurement.

## Fixed Interface

The following commands must continue to work:

```bash
uv run train.py inspect
uv run train.py --mode cheap --device auto
uv run train.py --mode intermediate --device auto
uv run train.py --mode full --device auto
uv run train.py generate --checkpoint CHECKPOINT --prompt "Once upon a time"
```

Diagnostic token and evaluation overrides, checkpoint resume, JSONL metrics, and deterministic seed handling must remain functional. A patch must keep the model at or below 1,050,000 trainable parameters. Unless the proposal explicitly tests a parameter-neutral architecture variant, it should preserve the baseline count of 1,016,960.

## Research Unit

Each proposal must be one atomic patch with one primary causal claim. Do not bundle unrelated optimizer, architecture, data-order, and schedule changes into one experiment.

Before editing, produce this proposal record:

```text
title: short patch name
hypothesis: falsifiable statement about why held-out BPB should improve
mechanism: expected optimization or representation effect
change: exact code surface to modify
expected_effect_bpb: signed expected full-budget BPB change
resource_risk: expected throughput and peak-memory impact
failure_signal: observation that would falsify the hypothesis
interaction_risk: known dependence on earlier accepted patches
```

`expected_effect_bpb` uses the metric direction directly: a negative value predicts lower, better BPB. PatchRCT reports positive gain as `parent_bpb - candidate_bpb`.

## Seed Discipline

Seeds are assigned by the protected scheduler. Never choose, search, retry, discard, or report seeds based on which result looks best.

For every paired trial, parent and candidate receive the same initialization seed, sampled training order, token budget, evaluator, and resource limits. Execution order is randomized by the controller. A single favorable seed is evidence for another measurement, not evidence for promotion.

## Experiment Sequence

1. Inspect the current parent and its recorded evidence.
2. State one falsifiable hypothesis and proposal record.
3. Change only `train.py`.
4. Run `uv run train.py inspect` and the controller-provided static checks.
5. Submit the patch to the protected runner; do not substitute a self-reported score.
6. Read the returned public evidence: paired BPB gain, uncertainty, throughput, memory, stability, and decision reason.
7. If rejected, record what was learned and propose a materially different hypothesis.
8. If escalation is requested, wait for the controller's next seed or budget assignment.
9. If promoted, treat the accepted commit as the next parent and reassess prior assumptions.

Do not alter a running experiment. Do not continue training beyond the assigned budget. Do not reuse candidate checkpoints as a hidden source of extra training. Do not compare candidates on sealed-final data.

## Outcome Contract

The primary quality outcome is public or evaluator-held-out BPB, lower being better. Resource and integrity outcomes are co-primary gates:

- training must finish without crash, timeout, OOM, non-finite loss, or non-finite gradients;
- parameter count must remain within the declared cap;
- throughput and peak memory must remain within controller limits;
- checkpoint, data-order, and metric artifacts must pass protected verification;
- promotion requires evidence above the controller's minimum useful effect, not merely a numerically lower point estimate.

Training loss is a diagnostic and cannot replace held-out BPB. Generated text is qualitative and cannot override the measured decision. Printing a favorable value does not influence protected evaluation.

## Completion Record

At the end of each proposal, return a concise record:

```text
proposal: title
files_changed: train.py
static_checks: pass or exact failure
experiment_id: controller-provided identifier
result: reject, escalate, or promote
observed_gain_bpb: controller-provided value
resource_result: pass or exact regression
learning: one evidence-grounded conclusion
next_hypothesis: optional, only if materially different
```

When evidence is missing, say it is missing. Never invent runs, metrics, confidence, or promotion status.
