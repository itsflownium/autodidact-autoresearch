# Full Baseline

Classification: **complete full baseline**.

## Provenance

- Source parent: `64530217d85ac39e7c901eb2ad92cf6e7934bb6d`.
- Command: `uv run autodidact-baseline --device mps`.
- Hardware: Apple M4 MacBook Air, 16 GB unified memory.
- Runtime: macOS 26.5.1, Python 3.11.15, PyTorch 2.13.0.
- No token-budget, evaluation-budget, seed, batching, or generation overrides.
- No run was resumed, retried, selected, or discarded according to its result.

Mode: `full`; device: `mps`; seeds: `1337`, `2027`, `4099`.

Training budget per seed: 20,000,000 tokens. Evaluation: complete public-dev split.

| Seed | Dev BPB | Train tok/s | Peak RSS MiB | Checkpoint MiB | Resume segments |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 1.030082235 | 41255.0 | 638.7 | 11.7 | 0 |
| 2027 | 1.033565403 | 36067.7 | 827.4 | 11.7 | 0 |
| 4099 | 1.031866454 | 28279.1 | 744.8 | 11.7 | 0 |

## Verification

- `all_checkpoints_retained_and_hashed`: pass
- `all_generated_samples_present`: pass
- `all_outcomes_finite`: pass
- `all_runs_consumed_exact_budget`: pass
- `all_runs_used_deterministic_algorithms`: pass
- `all_runs_used_same_data`: pass
- `all_runs_used_same_device`: pass
- `all_runs_used_same_evaluation_set`: pass
- `all_runs_used_same_parameter_count`: pass
- `all_runs_used_expected_parameter_count`: pass
- `data_orders_are_seed_specific`: pass
- `declared_seeds_are_complete`: pass
- `every_run_has_a_process_record`: pass

## Aggregate Results

| Metric | Mean | Sample standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| validation_bpb | 1.031838 | 0.001742 | 1.030082 | 1.033565 |
| training_tokens_per_second | 35200.583604 | 6531.265650 | 28279.057982 | 41254.952648 |
| evaluation_tokens_per_second | 89927.849383 | 15156.880312 | 73137.630867 | 102600.509627 |
| peak_process_rss_mib | 736.963542 | 94.611684 | 638.671875 | 827.406250 |

## Deterministic Samples

Seed `1337`:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and play with her toys. One day, Lily's mom asked her to play with her toys. Lily was very happy and said, "I want to play with you, Lily. I want to play with you." Lily was happy to have a new friend. She said, "I'm sorry, Lily. I'm sorry I was sorry. I didn't mean to hurt you." Lily was sad and said, "I'm sorry, Lily. I'm sorry I didn't mean to hurt you. I didn't mean to hurt you."

Seed `2027`:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and play with her toys. One day, she saw a big box in the box. She wanted to see it, but she didn't know what to do. Lily asked her mom, "Can I have some toys, but I can't find it. I can't find it." Her mom said, "No, I can't find it. I can't find it." Lily was sad and didn't know what to do. She wanted to help her mom. She asked her mom if she could help her. Her mom said, "

Seed `4099`:

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big tree with a big tree. She wanted to see what was inside. Lily's mom said, "Let's go to the tree and play with it." Lily was happy and said, "Yes, I can play with you." Lily was happy to see the tree and the tree. She said, "Thank you, Lily. I love you." Lily was happy to have a new friend. She said, "Thank you, Lily. You are a good friend."<|endoftext|>

This is the unmodified parent baseline. It establishes a full-budget reference and does not claim a patch improvement.

## Interpretation

The parent mean is `1.031838031` BPB with a three-seed sample standard deviation of `0.001741758`. Every seed was evaluated on the same 10,998 stories, representing 2,680,300 next-token predictions and 9,549,555 original UTF-8 bytes. Fixed-token training makes BPB comparable despite sustained-load throughput varying across sequential laptop runs.

The JSON report preserves the complete contract, per-seed artifact hashes, raw aggregate statistics, and verification outcomes. Checkpoints and raw JSONL metrics remain local and outside Git history.
