# Local baseline calibration

Mode: `cheap`; device: `mps`; training tokens per run: 2,000,000; evaluation tokens per run: 250,000.

Absolute tolerances: BPB `1e-07`; checkpoint tensors `1e-06`; checkpoint training loss `1e-07`.

| Run | Seed | Group | Dev BPB | Train tok/s | Peak process MiB | State hash |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| seed-104729 | 104729 | additional_seed | 1.784452571 | 37155.4 | 796.4 | `a2b40db12aa4` |
| seed-130363 | 130363 | additional_seed | 1.789929500 | 37088.5 | 795.6 | `a247832b884d` |
| seed-196613 | 196613 | additional_seed | 1.779577211 | 28746.7 | 753.0 | `6e58a5aa2e8a` |
| repeat-01 | 1337 | same_seed | 1.775246814 | 26742.2 | 696.2 | `c71b52403b79` |
| seed-7919 | 7919 | additional_seed | 1.782193275 | 23830.1 | 741.1 | `62ba7405274d` |
| repeat-02 | 1337 | same_seed | 1.775246826 | 20288.9 | 795.8 | `042157b1e4d2` |
| seed-2027 | 2027 | additional_seed | 1.783651626 | 20129.5 | 796.2 | `e013cc96844c` |
| seed-4099 | 4099 | additional_seed | 1.784596758 | 20246.1 | 796.9 | `00d564859946` |
| seed-155921 | 155921 | additional_seed | 1.788291528 | 20504.9 | 795.7 | `c549b6c10a6e` |
| repeat-03 | 1337 | same_seed | 1.775246814 | 20294.8 | 795.8 | `17085b3b727f` |

## Determinism checks

- `different_seeds_change_data_order`: pass
- `every_run_consumed_exact_budget`: pass
- `every_run_used_deterministic_algorithms`: pass
- `every_run_used_same_data_contract`: pass
- `every_run_used_same_device`: pass
- `every_run_used_same_evaluation_contract`: pass
- `every_run_used_same_parameter_count`: pass
- `outcomes_are_finite`: pass
- `same_seed_reproduces_checkpoint_within_tolerance`: pass
- `same_seed_reproduces_data_order`: pass
- `same_seed_reproduces_validation_bpb_within_tolerance`: pass
- `same_seed_validation_bpb_exact`: diagnostic drift
- `same_seed_validation_bpb_max_absolute_difference`: 1.25560402076e-08
- `same_seed_checkpoint_exact_hashes`: diagnostic drift
- `same_seed_model_max_absolute_difference`: 2.38418579102e-07
- `same_seed_optimizer_max_absolute_difference`: 5.12227416039e-09
- `same_seed_training_loss_max_absolute_difference`: 1.95312495066e-08

## Noise estimates

| Quantity | Estimate |
| --- | ---: |
| Same-seed execution BPB variance | 5.255138e-17 |
| Observed distinct-seed BPB variance | 0.000021662986 |
| Estimated seed BPB variance | 0.000021662986 |
| Estimated seed BPB standard deviation | 0.004654351 |

The seed component is the non-negative difference between distinct-seed and same-seed sample variances. It is a baseline planning estimate, not a model-quality claim.
