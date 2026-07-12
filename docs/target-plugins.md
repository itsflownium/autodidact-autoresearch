# Target plugin contract

A target plugin connects Autodidact's protected paired experiment runner to a user-selected model,
training stack, dataset, and evaluation metric. The researcher model remains separately configured
through `researcher.json`; changing Codex, Claude Code, or Hermes Agent does not change the target.

The bundled TinyStories target remains the fastest default for developing the autoresearch system.
External plugins can use larger models and different tasks because their target configuration sets
its own positive `max_parameter_count`.

## Files and trust boundary

The target repository contains three classes of files:

1. `editable_paths`: files the proposal agent may change in its one-commit patch.
2. `evaluator_path`: a Python adapter the agent cannot change.
3. The plugin and target JSON files: controller configuration the agent cannot change.

The trainer receives only `public_data_root`. The protected evaluator receives `data_root`, which
may include promotion or sealed splits. Use distinct directories and keep the evaluator-only root
unavailable to the researcher CLI. A plugin is trusted controller code: review it before use.

## Plugin JSON

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
      "--data-root", "{public_data_root}",
      "--seed", "{seed}",
      "--target-units", "{token_budget}",
      "--checkpoint", "{checkpoint}",
      "--metrics", "{metrics}"
    ],
    "evaluate": [
      "{python}", "{evaluator}", "evaluate",
      "--trainer", "{trainer}",
      "--checkpoint", "{checkpoint}",
      "--data-root", "{data_root}",
      "--split", "{split}",
      "--maximum-units", "{eval_tokens}"
    ]
  },
  "data_config_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "editable_paths": ["model/train.py", "model/layers.py"],
  "evaluator_path": "control/evaluate.py",
  "metric": {
    "direction": "higher",
    "name": "validation_accuracy",
    "objective_offset": 1.0,
    "objective_scale": 1.0
  },
  "plugin_id": "example.accuracy-target",
  "plugin_version": "1.0.0",
  "schema_version": 1,
  "tokenizer_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
  "trainer_path": "model/train.py"
}
```

The schema rejects unknown keys, unsafe repository paths, duplicate editable paths, an editable
evaluator, unsupported placeholders, and missing required placeholders. Each command starts with
`{python}` and a declared adapter path. Autodidact renders each element as one subprocess argument;
it does not concatenate or execute a shell string.

## Placeholders

All commands support `{python}`, `{trainer}`, `{evaluator}`, `{repository_root}`, and
`{parameter_cap}`. Training also supports `{public_data_root}`, `{stage}`, `{device}`, `{seed}`,
`{token_budget}`, `{batch_size}`, `{eval_batch_size}`, `{checkpoint}`, and `{metrics}`. Evaluation
also supports `{data_root}`, `{stage}`, `{split}`, `{device}`, `{seed}`, `{eval_tokens}`,
`{batch_size}`, and `{checkpoint}`. An `eval_tokens` value of `0` means the stage has no evaluation
unit ceiling.

`token_budget` and the legacy evidence names `tokens_seen` and `validation_bpb` represent generic
target units and a canonical lower-is-better objective for plugin targets. Raw metric identity and
direction remain in the protected evaluation artifact.

## Adapter events

The inspect command writes one JSON object to standard output:

```json
{"event":"target_inspection","parameter_count":1500000,"trainer_sha256":"..."}
```

The trainer writes JSON Lines to `{metrics}`. A successful run requires these final records:

```json
{"event":"target_training_config","seed":7,"target_units":2000000,"parameter_count":1500000,"data_config_sha256":"...","tokenizer_sha256":"..."}
{"event":"target_training_summary","seed":7,"target_units":2000000,"units_seen":2000000,"parameter_count":1500000,"data_config_sha256":"...","tokenizer_sha256":"...","data_order_sha256":"...","mean_train_loss":0.42}
```

The protected evaluator writes one JSON object to standard output:

```json
{"event":"target_evaluation","checkpoint_sha256":"...","trainer_sha256":"...","parameter_count":1500000,"metric_name":"validation_accuracy","metric_direction":"higher","metric_value":0.81,"evaluation_units":100000,"evaluation_seconds":2.5,"evaluation_units_per_second":40000.0,"peak_process_rss_bytes":500000000}
```

SHA-256 values are lowercase 64-character digests. The runner independently verifies trainer and
checkpoint hashes, parameter count, exact training budget, data-order commitment, metric identity,
finite outcomes, retained artifacts, worktree cleanliness, and resource constraints.

## Metric mapping

PatchRCT compares a canonical objective where lower is better. Plugins map raw metrics as follows:

- Lower-is-better: `objective = offset + scale * raw_metric`
- Higher-is-better: `objective = offset - scale * raw_metric`

`scale` must be positive, `offset` nonnegative, and every observed objective must remain
nonnegative. For accuracy in `[0, 1]`, use direction `higher`, offset `1`, and scale `1`; accuracy
`0.81` becomes objective `0.19`. Parent objective minus candidate objective is then a positive gain
when the candidate improves.

## Running locally or on a GPU host

Create a target file with `autodidact-target init`, run `autodidact-target doctor`, then pass the
same file to `autodidact-orchestrator --target-config ...`. For a remote GPU, place the repository,
plugin, public data, and protected data on that host and run the orchestrator there with device
`cuda` and execution location `gpu_host`. Autodidact does not provision machines, move datasets,
store provider credentials, or silently change the selected target.
