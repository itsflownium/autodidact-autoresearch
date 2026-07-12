# Autodidact configuration

Autodidact configures two different models. Keep them separate:

- The **researcher model** is the model used by Codex, Claude Code, or Hermes Agent to propose and
  implement changes.
- The **target model** is the model trained and measured by the paired autoresearch experiments.

Autodidact does not bundle or preload a trained 1M-parameter checkpoint into the research loop.
Every parent and candidate arm constructs and trains the target described by `train.py` under the
same declared seed, token budget, data contract, and device.

## Target model

Target schema version 1 accepts any architecture at or below the current 1,050,000-parameter cap
implemented behind Autodidact's protected `train.py` protocol. The included repository starts with
the 1,016,960-parameter TinyStories transformer, but that is a benchmark parent, not a hidden model
dependency.

Before starting a new campaign, put the target architecture and training recipe in `train.py` and
retain these commands:

```text
python train.py inspect
python train.py train ... --checkpoint-out PATH --metrics-file PATH
python train.py generate ...
```

The module must expose `build_model`, `model_config`, and checkpoint loading in the form expected by
the protected evaluator. `autodidact-target doctor` checks model construction and the parameter cap
without training the model.

Create the ignored local target configuration for a laptop:

```bash
uv run autodidact-target init \
  --name my-transformer \
  --data-root artifacts/data/tinystories-v1 \
  --device auto \
  --max-parameter-count 1050000

uv run autodidact-target doctor --repository-root .
```

`auto` selects the best locally available device supported by the trainer. Use `cpu`, `mps`, or
`cuda` to require one explicitly.

For an H100, clone or mount the repository and prepared data on the GPU host or container, then
create the target config and run the orchestrator there:

```bash
uv run autodidact-target init \
  --name my-transformer-h100 \
  --data-root /workspace/data/tinystories-v1 \
  --device cuda \
  --execution-location gpu_host \
  --max-parameter-count 1050000 \
  --estimated-accelerator-hour-usd 3.25

uv run autodidact-orchestrator \
  --target-config artifacts/control/target.json \
  --researcher-config artifacts/control/researcher.json \
  run
```

The user configures the GPU runtime, drivers, storage mount, and credentials. Autodidact does not
store cloud credentials or upload checkpoints. Running the orchestrator on the GPU host keeps the
protected evaluator, immutable data, parent arm, and candidate arm in one auditable environment.

Schema version 1 intentionally keeps `trainer_path` fixed to `train.py` and uses the protected
TinyStories BPB evaluator. Supporting arbitrary trainer paths, datasets, metrics, or distributed
launchers requires a future evaluator-plugin contract; merely accepting arbitrary shell commands
would weaken PatchRCT's evidence boundary.

## Researcher model

Choose the proposal agent and its model independently of the target:

```bash
uv run autodidact-agent bootstrap \
  --provider codex \
  --model RESEARCHER_MODEL_ID \
  --reasoning-effort high \
  --fix

uv run autodidact-agent bootstrap \
  --provider claude-code \
  --model RESEARCHER_MODEL_ID \
  --reasoning-effort high \
  --max-turns 40 \
  --max-budget-usd 5 \
  --fix

uv run autodidact-agent bootstrap \
  --provider hermes-agent \
  --backend-provider BACKEND_ID \
  --model RESEARCHER_MODEL_ID \
  --fix
```

To change the researcher model, rerun the command with the new model and `--force`. The selected
model ID is stored in `artifacts/control/researcher.json`; credentials remain in the provider CLI's
own credential store.

```bash
uv run autodidact-agent bootstrap \
  --provider codex \
  --model NEW_RESEARCHER_MODEL_ID \
  --reasoning-effort high \
  --force \
  --fix
```

Bootstrap and doctor perform only version and help-capability probes. They do not spend inference
tokens. Before every real proposal, the adapter repeats those checks and refuses to call the model
if required flags have disappeared.

```bash
uv run autodidact-agent doctor --fix
```

`--fix` can select another compatible executable, restrict config permissions, run documented npm
installation or provider upgrade commands, and then probe again. It never executes a shell string.
A missing Hermes installation is reported with the official installer URL instead of automatically
executing a downloaded script.

CLI compatibility and model availability are different checks. Help output can prove that an
installed CLI supports Autodidact's invocation contract, but an authenticated provider may still
reject an unavailable or misspelled model ID. That provider error is retained in the research
transcript; Autodidact cannot safely replace the user's chosen model with a different billed model.

## Configuration files

Both files are local runtime control artifacts and should remain outside Git history:

| File | Purpose |
| --- | --- |
| `artifacts/control/researcher.json` | Proposal-agent CLI, model, budgets, and timeout |
| `artifacts/control/target.json` | Target name, data root, device, parameter cap, and cost estimate |

Neither file contains API keys. Provider authentication stays with the provider CLI, and GPU/cloud
authentication stays with the user's host environment.
