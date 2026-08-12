# Target plugin and RL/RLVR protocol

A target plugin connects Autodidact to user-supplied training and protected evaluation commands.
The plugin is target code, not a bundled target: Autodidact ships no model, dataset, environment,
verifier, or RL algorithm.

## Trust boundary

The target repository has three file classes:

1. `editable_paths`: the only files the researcher may inspect or modify.
2. `evaluator_path`: protected adapter for inspection and evaluation.
3. Plugin, target config, protected data/tasks, and Autodidact control state: not editable.

For RL and RLVR, `rl.algorithm_paths` must be a subset of `editable_paths`. This is how the chosen
researcher gets authority to create, replace, or tune the algorithm while the reward contract stays
protected.

Commands are argument arrays, not shell strings. Autodidact renders each item as one subprocess
argument and requires `{python}` followed by the declared trainer or evaluator entry point.

## Schema version 2

```json
{
  "commands": {
    "inspect": [
      "{python}", "{evaluator}", "inspect",
      "--trainer", "{trainer}",
      "--parameter-cap", "{parameter_cap}"
    ],
    "train": [
      "{python}", "{trainer}", "train",
      "--public-root", "{public_data_root}",
      "--device", "{device}",
      "--seed", "{seed}",
      "--budget", "{training_budget}",
      "--checkpoint", "{checkpoint}",
      "--metrics", "{metrics}"
    ],
    "evaluate": [
      "{python}", "{evaluator}", "evaluate",
      "--trainer", "{trainer}",
      "--checkpoint", "{checkpoint}",
      "--protected-root", "{data_root}",
      "--split", "{split}",
      "--maximum-units", "{eval_tokens}"
    ]
  },
  "data_config_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "editable_paths": ["policy/train.py", "policy/algorithm.py", "policy/model.py"],
  "evaluator_path": "control/evaluate.py",
  "metric": {
    "direction": "higher",
    "name": "verified_reward",
    "objective_offset": 1.0,
    "objective_scale": 1.0
  },
  "plugin_id": "example.rlvr-target",
  "plugin_version": "1.0.0",
  "rl": {
    "algorithm_paths": ["policy/algorithm.py"],
    "budget_unit": "tokens",
    "paradigm": "rlvr",
    "reward_maximum": 1.0,
    "reward_minimum": 0.0,
    "reward_source": "verifier",
    "schema_version": 1
  },
  "schema_version": 2,
  "tokenizer_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
  "trainer_path": "policy/train.py"
}
```

Set `rl` to `null` for a non-RL target. Schema-version-1 plugins remain readable and are interpreted
as non-RL targets.

The legacy field names `data_config_sha256` and `tokenizer_sha256` are retained in evidence schema
version 2 for compatibility. For a non-language target, use them as immutable commitments to the
training-source contract and input-encoding/preprocessing contract respectively. Do not insert
arbitrary hashes: derive them from canonical target metadata and update the plugin version when the
contract changes.

## RL contract

| Field | Protected meaning |
| --- | --- |
| `paradigm` | `rl` or `rlvr` |
| `reward_source` | `environment`, `reward_model`, `verifier`, or `hybrid` |
| `budget_unit` | Unit that `{training_budget}` and `units_seen` represent |
| `reward_minimum`, `reward_maximum` | Finite inclusive range for train and evaluation rewards |
| `algorithm_paths` | Research-editable files implementing the current algorithm |

An RLVR contract requires `verifier` or `hybrid` reward. The plugin metric must be
higher-is-better. The contract intentionally has no `algorithm` field: the agent can implement
GRPO, PPO, REINFORCE, another published method, or a custom method in `algorithm_paths`.

The mutable trainer reports an `algorithm_id` for attribution. The protected evaluator does not
trust that identifier as evidence and independently computes the declared reward.

## Placeholders

All commands support:

`{python}`, `{trainer}`, `{evaluator}`, `{repository_root}`, `{parameter_cap}`, `{device}`

Training additionally supports:

`{public_data_root}`, `{stage}`, `{seed}`, `{training_budget}`, `{token_budget}`, `{batch_size}`,
`{eval_batch_size}`, `{checkpoint}`, `{metrics}`

Evaluation additionally supports:

`{data_root}`, `{stage}`, `{split}`, `{seed}`, `{eval_tokens}`, `{batch_size}`, `{checkpoint}`

Use `{training_budget}` for new plugins. `{token_budget}` remains a compatibility alias. An
`eval_tokens` value of `0` means the stage has no evaluation-unit ceiling.

## Inspect event

The protected inspect command writes one JSON object to stdout:

```json
{
  "event": "target_inspection",
  "parameter_count": 125000000,
  "trainer_sha256": "<sha256 of the supplied trainer>"
}
```

Autodidact rejects a missing or mismatched trainer hash and any parameter count above the target
cap.

## Training events

The trainer writes JSON Lines to `{metrics}`. A successful generic run ends with:

```json
{"event":"target_training_config","seed":11,"target_units":2000000,"parameter_count":125000000,"data_config_sha256":"...","tokenizer_sha256":"..."}
{"event":"target_training_summary","seed":11,"target_units":2000000,"units_seen":2000000,"parameter_count":125000000,"data_order_sha256":"...","mean_train_loss":0.42}
```

For RL or RLVR, both records also include the protected `training_paradigm` and `budget_unit`. The
summary additionally requires:

```json
{
  "algorithm_id": "custom-policy-objective-v3",
  "budget_unit": "tokens",
  "event": "target_training_summary",
  "mean_train_loss": 0.18,
  "mean_train_reward": 0.57,
  "rollout_valid_fraction": 0.99,
  "train_reward_standard_deviation": 0.21,
  "training_paradigm": "rlvr",
  "units_seen": 2000000
}
```

Optional finite diagnostics are `policy_loss`, nonnegative `kl_divergence`, and nonnegative
`policy_entropy`. The trainer must consume exactly the assigned budget. `data_order_sha256` commits
to sampled tasks, examples, prompts, or environment episodes in their actual order.

## Protected evaluation event

The evaluator writes one JSON object to stdout:

```json
{
  "checkpoint_sha256": "...",
  "evaluation_seconds": 12.4,
  "evaluation_units": 4096,
  "evaluation_units_per_second": 330.3,
  "event": "target_evaluation",
  "metric_direction": "higher",
  "metric_name": "verified_reward",
  "metric_value": 0.61,
  "parameter_count": 125000000,
  "reward_source": "verifier",
  "reward_standard_deviation": 0.16,
  "trainer_sha256": "...",
  "training_paradigm": "rlvr",
  "verifier_coverage": 1.0
}
```

RL requires reward standard deviation. RLVR also requires verifier coverage in `[0, 1]`. The
metric value must remain within the declared reward range. The runner independently verifies the
checkpoint, trainer, metric, parameter count, resource measurements, and worktree cleanliness.

## Metric mapping

PatchRCT uses one canonical lower-is-better objective:

- lower-is-better raw metric: `objective = offset + scale * raw_metric`
- higher-is-better raw metric: `objective = offset - scale * raw_metric`

`scale` is positive and `offset` is nonnegative. Every observed canonical objective must remain
nonnegative. With reward in `[0, 1]`, offset `1` and scale `1` map reward `0.61` to objective `0.39`.
The paired gain is `parent objective - candidate objective`, so positive always means improvement.

## Operational rules

- Review the plugin and evaluator as trusted controller code before a campaign.
- Keep protected tasks and verifier-only data unavailable to the researcher process.
- Make target commands deterministic for a fixed seed where the runtime permits it.
- Report failures honestly; never write a successful summary before the checkpoint is durable.
- Keep credentials outside command templates and source control.
- Start a new campaign whenever reward semantics, preprocessing, evaluator, or target contract
  changes.

See the complete examples under [`examples/`](../examples/).
