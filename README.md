# Autodidact

Autodidact is a model-agnostic autoresearch control plane. A research agent proposes and implements
one code change, Autodidact trains the unchanged parent and candidate under matched conditions, and
PatchRCT decides whether the evidence is strong enough to promote the candidate.

The repository contains no model, dataset, checkpoint, or fixed reinforcement-learning algorithm.
You connect your own supervised, RL, or RLVR target through a protected target plugin. The
researcher can be Codex, Claude Code, Hermes Agent, or a custom command adapter.

Project site: [researchautodidact.vercel.app](https://researchautodidact.vercel.app/)

## What it controls

```mermaid
flowchart LR
    A["Research direction and prior evidence"] --> B["Researcher agent"]
    B --> C["One patch within editable_paths"]
    C --> D["Paired parent and candidate runs"]
    D --> E["Protected evaluator or verifier"]
    E --> F["PatchRCT decision"]
    F -->|promote| G["New accepted parent"]
    F -->|escalate| D
    F -->|reject| A
    G --> A
    E --> H["Append-only evidence ledger"]
```

- **Researcher boundary:** the agent may inspect and modify only the plugin's `editable_paths`.
- **Target boundary:** the user supplies training, data or tasks, checkpoints, and device handling.
- **Evaluation boundary:** the evaluator, verifier, target contract, seeds, and budgets are protected.
- **Decision boundary:** the agent proposes code but cannot grade or promote its own patch.
- **Recovery boundary:** transcripts, reservations, runs, decisions, and lineage are durably recorded.

## RL and RLVR

An RL target declares the stable experimental facts: `rl` or `rlvr`, reward source, reward range,
budget unit, and which editable files implement the algorithm. It deliberately does **not** declare
a fixed algorithm.

For example, a plugin can expose `policy/algorithm.py` in both `editable_paths` and
`rl.algorithm_paths`. Codex, Claude Code, Hermes Agent, or another configured researcher may then
replace GRPO with PPO, implement a custom objective, change advantage estimation, or tune rollout
logic as its one proposed patch. Autodidact records the trainer-reported `algorithm_id` for each
run but does not restrict its value.

The protected contract still prevents the researcher from changing reward semantics, the verifier,
held-out tasks, seed assignment, or promotion thresholds. For RLVR, the reward source must include a
verifier, and protected evaluation must report verifier coverage and reward variance.

## Install

Python 3.11 through 3.13 and Git are required. Install from the repository with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/itsflownium/autodidact-autoresearch.git
cd autodidact-autoresearch
uv sync --locked
```

Autodidact can also be installed into a separate target repository:

```bash
uv add "autodidact-autoresearch @ git+https://github.com/itsflownium/autodidact-autoresearch.git"
```

Runtime configuration and evidence are written under `artifacts/` and should stay outside Git.
Credentials remain in the research-agent CLI or GPU host; Autodidact does not store API keys.

## 1. Configure a researcher

Choose one proposal agent independently from the model being researched.

### Codex

```bash
uv run autodidact-agent bootstrap \
  --provider codex \
  --model YOUR_CODEX_MODEL \
  --reasoning-effort high \
  --fix
```

### Claude Code

```bash
uv run autodidact-agent bootstrap \
  --provider claude-code \
  --model YOUR_CLAUDE_MODEL \
  --reasoning-effort high \
  --max-turns 40 \
  --fix
```

### Hermes Agent

```bash
uv run autodidact-agent bootstrap \
  --provider hermes-agent \
  --backend-provider YOUR_HERMES_BACKEND \
  --model YOUR_RESEARCHER_MODEL \
  --fix
```

Then verify the executable and local configuration without making an inference call:

```bash
uv run autodidact-agent doctor --fix
```

`doctor --fix` can locate another compatible executable, repair config permissions, and run known
package-manager repairs. It fails before inference when a CLI version no longer supports the flags
Autodidact needs.

## 2. Connect a target

Add three things to the Git repository being researched:

1. Editable trainer and algorithm files.
2. A protected evaluator adapter that implements the JSON event protocol.
3. A protected target-plugin JSON file that declares commands, paths, metric, and optional RL data.

Start from [`examples/rlvr-target-plugin.example.json`](examples/rlvr-target-plugin.example.json)
for RLVR or [`examples/target-plugin.example.json`](examples/target-plugin.example.json) for another
training paradigm. These are contracts only; they do not include a model or task implementation.

Create the local target configuration:

```bash
uv run autodidact-target init \
  --name my-rlvr-target \
  --trainer-path policy/train.py \
  --plugin-spec control/target-plugin.json \
  --public-data-root /datasets/my-task/public \
  --data-root /datasets/my-task/protected \
  --device cuda \
  --execution-location gpu_host \
  --max-parameter-count 8000000000

uv run autodidact-target doctor --repository-root .
```

`public-data-root` is visible to training. `data-root` is passed only to the protected evaluator.
They must be different directories. `device` is an opaque value passed to your adapters, so a
target may support `cpu`, `mps`, `cuda`, a local consumer GPU, or a custom remote runtime.
Autodidact does not provision or SSH into that runtime.

## 3. Run a campaign

Keep a generic research contract such as [`program.md`](program.md) in the target repository, or
pass a different file with `--program`.

```bash
uv run autodidact-orchestrator \
  --repository-root . \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  --program program.md \
  --decision-mode patch_rct \
  initialize \
  --campaign-id rlvr-pilot-001 \
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

Run one proposal first and inspect status before increasing the limit:

```bash
uv run autodidact-orchestrator \
  --repository-root . \
  --researcher-config artifacts/control/researcher.json \
  --target-config artifacts/control/target.json \
  status
```

Initialization pins hashes of the researcher config, target config, program, optional execution
queue, decision mode, and campaign ceilings. Repeating `run` resumes recorded work instead of
duplicating a completed researcher call or protected experiment.

## PatchRCT

PatchRCT means **Patch Randomized Controlled Trial**. For each proposed patch it compares:

- a parent arm at the currently accepted Git commit; and
- a candidate arm with exactly one agent-authored patch.

Both arms use predetermined seeds, matched budgets, the same protected metric, randomized execution
order, and resource gates. A positive gain means the candidate's canonical objective is lower than
the parent's. The Bayesian posterior estimates the probability that this gain exceeds the
proposal's predeclared minimum useful effect. PatchRCT can reject, request more paired evidence, or
promote only after full-stage evidence clears the configured probability and regression gates.

This is stricter than a single noisy keep/discard result. It costs more per difficult decision, so
Autodidact also supports a scout mode and a separately calibrated downstream reward estimator.

## Downstream reward estimation

With `--reward-calibration-labels N --use-downstream-allocation`, early paired measurements become
features for a protected Bayesian learning-curve model. After enough candidates also receive full
labels, the model can stop clearly weak candidates, gather more intermediate evidence, or allocate a
full test. Deterministic audits preserve labels needed to detect estimator drift.

The estimator allocates evaluation compute; it never promotes a patch. Final promotion still uses
direct PatchRCT evidence from the protected metric.

## Evidence and reporting

The append-only SQLite ledger records proposal claims, candidate commits, run contracts, artifacts,
paired effects, posterior decisions, resource measurements, and accepted lineage. The sealed
workflow freezes accepted lineages and evaluates them on untouched tasks after research ends.

- [Configuration](docs/configuration.md)
- [Target plugin and RL/RLVR protocol](docs/target-plugins.md)
- [Sealed evaluation](docs/sealed-evaluation.md)
- [Three-arm comparison study](docs/three-arm-study.md)

## Development

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -m compileall -q autodidact tests
uv build
```

The tests use synthetic target plugins and do not download a dataset or train a bundled model.
