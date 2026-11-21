# Pre-LSTM Baseline (2026-11-18)

This snapshot records the current non-recurrent policy and the result of one 10M-step fine-tuning attempt before trying LSTM memory. It is a comparison baseline, not a claim that the fine-tuned model is better.

## Models

- `ExplorerAgent-10000032.onnx`: reference policy.
- `ExplorerAgent-10000113.onnx`: initialized from `visible_tiles_1m_100area_20261115_r2` and fine-tuned for approximately 10M steps; its run configuration is in `results/collision_relay_finetune_20261118_10m_r1/configuration.yaml`.

## Evaluation

Both policies were evaluated on seeds 1–1000 with four parallel areas, time scale 5, and 20x20 maps with 20 waypoints. Action tracing was disabled. The paired JSON summaries and failure records are stored beside this file.

| Layout | Reference successes | Fine-tuned successes |
| --- | ---: | ---: |
| PathCorridor | 637/1000 (63.7%) | 550/1000 (55.0%) |
| RoomsAndCorridors | 598/1000 (59.8%) | 377/1000 (37.7%) |

All recorded failures ended at the step limit. Since action tracing was disabled, these results do not identify the exact failing behavior. The fine-tuned model is a regression in these evaluations and should not replace the reference policy as champion.

## Interpretation

The fine-tuning run used `resume: false` and `time_scale: 20`; evaluation used time scale 5. This is not a controlled continuation under matching conditions. Compare future recurrent-policy experiments against the reference model under the same map, seed, parallelism, and time-scale settings.
