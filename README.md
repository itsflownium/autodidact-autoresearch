# autodidact

The idea: give an AI research agent a small but real language-model training setup and let it improve the model autonomously. The agent proposes a focused change, trains it against the current parent under matched conditions, and receives a decision: reject it, gather more evidence, or promote it.

Autodidact starts with a **1,016,960-parameter transformer** that can run locally on Apple Silicon. Larger studies can move to a single H100 without changing the research protocol. The target is not a frontier language model. It is a controlled laboratory for asking a more basic question:

> Can an autonomous research loop distinguish genuine model improvements from lucky training runs?

Autodidact builds on the compact workflow introduced by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), then adds **PatchRCT**, paired experiments, hidden evaluation, and Bayesian downstream-reward estimation.

> **Status:** design and baseline implementation are in progress. No model-quality result is claimed yet.

## How it works

The core loop is deliberately small:

```text
research agent proposes an atomic patch + falsifiable claim
                              |
                              v
                 integrity and parameter checks
                              |
                              v
             parent and patch run on matched trials
                              |
                              v
          patch effect + downstream reward are estimated
                              |
                              v
                  reject / escalate / promote
                              |
                              v
             accepted patch becomes the new parent
```

The research agent proposes changes, while the protected experiment runner and PatchRCT system evaluate them. The agent does not grade its own work.

The initial implementation keeps the same useful boundary as autoresearch:

- **`prepare.py`** — fixed data preparation, tokenizer, dataloader, experiment budgets, and evaluation. The agent cannot modify it.
- **`train.py`** — the transformer, optimizer, and training loop. This is the only file the agent may modify.
- **`program.md`** — instructions and research context supplied to the agent.
- **`autodidact/`** — the protected PatchRCT harness, downstream-reward estimator, experiment ledger, and promotion rules.

## The 1M-parameter model

The baseline is a decoder-only, GPT-style transformer trained from scratch on [TinyStories](https://arxiv.org/abs/2305.07759). TinyStories has constrained language and subject matter, which makes meaningful language modelling possible at this scale.

| Component | Baseline |
| --- | ---: |
| Vocabulary | 1,536-token BPE |
| Context length | 256 tokens |
| Transformer layers | 4 |
| Model width | 128 |
| Attention heads | 4 |
| Head dimension | 32 |
| MLP width | 512 |
| Normalization | Pre-LayerNorm |
| Activation | GELU |
| Position representation | Learned embeddings |
| Input/output embeddings | Tied |
| Linear biases | Disabled |
| Initial dropout | 0 |
| Trainable parameters | **1,016,960** |
| Autoresearch limit | **1,050,000** |

The starting training recipe uses AdamW, a cosine learning-rate schedule, gradient clipping, and a 16,384-token batch. Cheap, intermediate, and full experiments train for 2M, 6M, and 20M tokens respectively. These are baseline choices, not protected truths: the research agent may change the architecture, optimizer, batching, and training logic while remaining inside the parameter and evaluation constraints.

## What is BPB?

The primary quality metric is **validation bits per byte**, or `val_bpb`. It is the model's negative log-probability on held-out text, normalized by the number of original UTF-8 bytes:

```text
bpb = total negative log2 probability / number of original text bytes
```

Lower is better. A lower BPB means the model predicts the held-out text more efficiently. Normalizing by bytes also makes the metric less dependent on how the tokenizer divides text into tokens.

For a parent and candidate patch, Autodidact uses a positive-gain convention:

```text
patch gain = parent_bpb - patch_bpb
```

## What is PatchRCT?

PatchRCT is an RCT-inspired validation and promotion protocol for code changes. It is not a model and it is not a research agent.

For each candidate:

- The **parent** is the current accepted training implementation.
- The **treatment** is the parent plus one proposed patch.
- A **block** is a matched seed, initialization, data order, budget, and evaluator.
- The **outcome** includes BPB, throughput, memory, and stability.

Parent and patch are both trained for every selected seed. Their execution order is randomized to reduce hardware and thermal drift. Because the intended code patch is the only systematic difference, the paired result gives a much cleaner estimate of that patch's effect than one isolated training run.

An illustrative trial might look like this:

```text
seed 11: parent 1.421  patch 1.417  gain +0.004
seed 23: parent 1.419  patch 1.416  gain +0.003
seed 37: parent 1.420  patch 1.421  gain -0.001
```

PatchRCT validates candidates in stages:

1. Confirm that the patch is atomic, only allowed files changed, the model remains under the parameter cap, and training finishes safely.
2. Run one cheap paired trial.
3. Run additional paired seeds when the first result is promising.
4. Estimate whether the early advantage is likely to survive the full training budget.
5. Confirm strong candidates on evaluator-only data and seeds.
6. Reject, request more evidence, or promote the patch.
7. Periodically test whether promoted patches still help when combined.

A patch is promoted only when the probability of exceeding the predeclared minimum useful effect is high enough and the runtime, memory, parameter, and stability constraints all pass. The resulting claim is intentionally narrow: the patch improved this parent, on this task, under these tested conditions.

## Estimating downstream reward

A patch can look good after 2M tokens and become neutral or harmful after 20M. Autodidact therefore treats the short experiment as evidence about the objective, not the objective itself.

For this project, **downstream reward** is the patch's held-out BPB improvement after the full budget, measured on unseen seeds and data and subject to the resource constraints. The estimator observes early signals such as:

- paired BPB differences at several checkpoints;
- learning-curve slope and area;
- disagreement across seeds;
- training-versus-development gap;
- throughput and peak memory;
- loss spikes, failures, and parameter changes.

The first 40 valid patches receive both short and full evaluations. Those completed experiments train a Bayesian learning-curve model that predicts a distribution rather than a single final score:

```text
expected full-budget gain:  +0.0031 bpb
probability gain is useful: 87%
```

The uncertainty drives compute allocation. Clearly poor candidates die early, uncertain candidates receive another seed or a longer run, and strong candidates proceed to held-out confirmation. A random sample of rejected patches is still fully evaluated so that missed improvements and estimator bias remain measurable.

The downstream estimator does not replace PatchRCT. It helps PatchRCT decide **which experiment would be most useful next**.

## How this extends autoresearch

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) provides the foundation: one editable training file, a fixed experiment budget, a single comparable metric, a Git lineage, and a simple keep-or-discard loop.

Autodidact keeps those strengths and changes the promotion question:

```text
autoresearch: did this run improve?
autodidact:   did this patch cause a useful, repeatable improvement?
```

The first study will compare three systems under matched research budgets:

1. A Karpathy-style greedy keep/discard loop.
2. PatchRCT with direct full-budget confirmation.
3. PatchRCT with Bayesian downstream-reward estimation.

This separates the benefit of paired patch validation from the additional compute savings of learning-curve prediction.

## Local first, one H100 later

The baseline is small enough to train locally on an Apple Silicon laptop using PyTorch MPS. Local execution is the default for model development, baseline calibration, seed-noise measurement, and the first autonomous research pilot.

A single H100 can later be used for more seeds, longer runs, repeated research lineages, and cross-hardware confirmation. No multi-GPU setup is required.

Fixed-token results provide the cleanest comparison across hardware. Fixed-time results are platform-specific, so the laptop and H100 receive separate baseline timing calibrations.

## Design choices

- **One mutable file.** Small diffs are easier to attribute, audit, revert, and compose.
- **A hard parameter budget.** Improvements cannot come from silently making the model larger.
- **Paired evidence.** Parent and patch share seeds, data order, evaluator, and budget.
- **Sequential compute.** Bad ideas fail cheaply; uncertainty earns more experimentation.
- **A separate evaluator.** The proposer cannot edit the tests or promote itself.
- **A sealed final result.** The final data and seed set are not reused during research.
- **An evidence ledger.** Every claim, diff, run, prediction, decision, and accepted commit is retained.

## Planned repository structure

```text
prepare.py              fixed data and evaluation code
train.py                model and training code; agent-editable
program.md              research instructions
autodidact/              PatchRCT, orchestration, reward model, and ledger
experiments/             manifests and machine-readable results
assets/                  final figures
```

## Roadmap

- [ ] Train and verify the 1,016,960-parameter TinyStories baseline.
- [ ] Measure seed noise and execution noise locally.
- [ ] Implement protected paired parent-versus-patch experiments.
- [ ] Implement PatchRCT promotion, rejection, and escalation gates.
- [ ] Collect 40 full-budget patch labels and calibrate downstream prediction.
- [ ] Run the three-arm, 50-proposal local pilot.
- [ ] Expand to repeated 100-proposal studies.
- [ ] Confirm selected findings on one H100.
- [ ] Publish the sealed result graph and experiment ledger.

## License

MIT. See [LICENSE](LICENSE).

## Results

No result is reported before the sealed evaluation. The final graph will compare hidden BPB, false promotions, and compute per confirmed improvement across the three research systems.

<!-- Replace this comment after the sealed evaluation:
![Autodidact results](assets/results.png)
-->
