# Configuration

Autodidact separates the **researcher** from the **target**. The researcher is the coding agent
that authors one patch. The target is the model, environment, training process, data or tasks, and
metric being improved. Either side can change without silently changing the other.

The source distribution contains only the autoresearch control plane. It does not contain a
TinyStories model, a one-million-parameter model, a dataset, a checkpoint, or a default RL
algorithm.

## Researcher configuration

Create `artifacts/control/researcher.json` with a native adapter:

```bash
# Codex
uv run autodidact-agent bootstrap \
  --provider codex \
  --model YOUR_CODEX_MODEL \
  --reasoning-effort high \
  --fix

# Claude Code
uv run autodidact-agent bootstrap \
  --provider claude-code \
  --model YOUR_CLAUDE_MODEL \
  --reasoning-effort high \
  --max-turns 40 \
  --fix

# Hermes Agent
uv run autodidact-agent bootstrap \
  --provider hermes-agent \
  --backend-provider YOUR_BACKEND \
  --model YOUR_RESEARCHER_MODEL \
  --fix
```

The configured agent receives the research program, prior public evidence, target summary, and an
exact editable-path allowlist. It works in an isolated candidate worktree. The same target works
with all three adapters.

Change the researcher by running bootstrap again with `--force`. This starts no training and does
not change the target contract. Credentials stay in the provider's own CLI configuration.

```bash
uv run autodidact-agent doctor --fix
```

Doctor probes the executable and required flags before inference. `--fix` attempts known local
installation and permission repairs, then probes again. It cannot guarantee compatibility with
future third-party CLI releases, but it fails with a specific diagnostic before spending an
inference call.

## Target configuration

The target is described by a local target JSON file and a protected plugin JSON file. The target
files may describe any parameter count and any supported device because the repository has no
built-in target.

```bash
uv run autodidact-target init \
  --name my-target \
  --trainer-path policy/train.py \
  --plugin-spec control/target-plugin.json \
  --public-data-root /datasets/my-target/public \
  --data-root /datasets/my-target/protected \
  --device cuda \
  --execution-location gpu_host \
  --estimated-accelerator-hour-usd 1.50 \
  --max-parameter-count 8000000000

uv run autodidact-target doctor \
  --config artifacts/control/target.json \
  --repository-root .
```

Target fields:

| Field | Meaning |
| --- | --- |
| `trainer_path` | Repository-relative entry point that the researcher may edit |
| `plugin_spec_path` | Protected command, metric, path, and optional RL contract |
| `public_data_root` | Data, tasks, or environment inputs visible during training |
| `data_root` | Separate protected inputs visible only to evaluation |
| `device` | Opaque device name passed to the target commands |
| `execution_location` | Audit label: `local` or `gpu_host` |
| `max_parameter_count` | Hard cap checked by protected inspection |
| `estimated_accelerator_hour_usd` | Optional accounting rate, never a cloud credential |

The public and protected roots must exist and must differ. Use separate permissions or mounts when
the threat model requires the researcher process to be unable to read the protected root.

For a local GPU, set the device string expected by the trainer, such as `cuda`. For a remote GPU,
clone or mount the target repository and data on that host, install Autodidact there, and run the
orchestrator there. Autodidact invokes the configured command locally; it does not provision a VM,
upload data, or establish SSH sessions.

## Custom RL algorithms

RL is configured in the plugin, not in the researcher adapter. The protected `rl` object declares
only experimental invariants:

- `paradigm`: `rl` or `rlvr`;
- `reward_source`: `environment`, `reward_model`, `verifier`, or `hybrid`;
- `budget_unit`: the unit consumed exactly by training;
- `reward_minimum` and `reward_maximum`; and
- `algorithm_paths`: files that contain the mutable algorithm.

Every algorithm path must also be in `editable_paths`. Therefore the selected researcher can
implement or replace the algorithm as a normal proposal. There is no algorithm enum and no forced
GRPO implementation. The protected trainer must emit an `algorithm_id` describing what actually
ran, so evidence remains attributable after the agent changes it.

Reward semantics, verifier code, protected tasks, metric transforms, and budgets remain outside the
editable allowlist. This lets an agent research the learning algorithm without letting it redefine
success.

## Campaign configuration

Initialize once, then use the same control files for every `run` or `status` command:

```bash
uv run autodidact-orchestrator \
  --repository-root . \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  --program program.md \
  --decision-mode patch_rct \
  initialize \
  --campaign-id pilot-001 \
  --max-proposals 20 \
  --max-wall-seconds 86400 \
  --max-researcher-tokens 20000000 \
  --max-training-tokens 400000000 \
  --max-compute-seconds 86400

uv run autodidact-orchestrator \
  --repository-root . \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  run --max-new-proposals 1
```

`max-researcher-tokens` limits coding-agent usage. `max-training-tokens` is the historical internal
name for the campaign's target-training units; an RL plugin receives each stage value through
`{training_budget}` and reports its declared `budget_unit`.

Initialization pins hashes of the program, researcher config, target config, optional queue, and
campaign limits. Recovery rejects drift. To change a researcher, target, metric, data contract, or
device, initialize a new campaign rather than editing a running campaign's pinned files.

## Optional downstream allocation

```bash
uv run autodidact-orchestrator \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  initialize \
  --campaign-id calibrated-001 \
  --max-proposals 60 \
  --max-wall-seconds 604800 \
  --max-researcher-tokens 60000000 \
  --max-training-tokens 9000000000 \
  --max-compute-seconds 604800 \
  --reward-calibration-labels 40 \
  --use-downstream-allocation
```

The calibration phase collects early features plus direct full-stage labels. Once ready, a Bayesian
learning-curve model helps allocate later full tests. It never replaces protected evaluation or the
PatchRCT promotion decision.

## Local control files

| Path | Purpose |
| --- | --- |
| `artifacts/control/researcher.json` | Researcher CLI, selected model, and inference limits |
| `artifacts/control/target.json` | Target plugin, roots, device, cap, and cost label |
| `artifacts/state/campaign.sqlite3` | Recoverable campaign state |
| `artifacts/ledger/experiments.sqlite3` | Append-only experiment evidence |

These files contain no API keys but can reveal local paths and experiment history. Keep them out of
the source repository unless explicitly redacted for publication.
