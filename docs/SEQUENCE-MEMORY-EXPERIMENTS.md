# Sequence Memory and Agent Landmarks

This branch adds recurrent-memory options and agent-selected landmark observations. The validation snapshot below records the current training and inference evidence.

## Agent-selected landmarks

Each ExplorerAgent has a mixed action space: two continuous movement values plus one discrete branch (`0` = do not place, `1` = place at the current cell). The agent can place at most one landmark per visited cell and 64 per episode. Landmarks are cleared on episode reset. They add no reward or penalty. A BufferSensor presents each landmark as agent-relative X/Z offsets and normalized age. Its attention encoder processes this set alongside the existing fixed-size map observations.

The `enableLandmarkActions` and `exposeLandmarkObservations` fields can independently disable landmark writing and landmark observations later for ablation runs.

## LSTM option

Use `Assets/config/trainer_lstm.yaml` for the ML-Agents built-in LSTM. The current `trainer.yaml` remains the feed-forward comparison configuration.

## GTrXL option

Use `Assets/config/trainer_gtrxl.yaml` for the project-specific `gtrxl_ppo` plugin. It implements a GTrXL-style PPO actor/critic with segment memory, relative-position attention, and gated residual blocks. It reuses ML-Agents observation encoders, including BufferSensor processing. The trainer uses a 16-step cached context per layer and a 64-step PPO sequence.

Install the plugin into the ML-Agents virtual environment before using that configuration:

```powershell
.\install-mlagents-gtrxl.ps1
.\.venv-mlagents\Scripts\mlagents-learn.exe Assets/config/trainer_gtrxl.yaml --torch-device=cuda --run-id=<name>
```

On another platform, activate its ML-Agents virtual environment and run `python -m pip install -e ./mlagents-gtrxl`.

The plugin pins PyTorch 2.8.0 because newer PyTorch releases have known ONNX export incompatibilities with this ML-Agents line. The install script for a pre-existing environment will enforce that version.

For the built-in LSTM configuration, replace the YAML path with `Assets/config/trainer_lstm.yaml` and choose a new run ID. Existing ONNX models do not match the updated action/observation contract; train and export a new model before inference testing.

## Validation snapshot (Windows, 2026-11-22)

- The custom plugin started with ML-Agents 1.1.0 and PyTorch 2.8.0+cu128, resumed the 50,870-step checkpoint, performed PPO updates, and exported ONNX models.
- The newest saved training checkpoint reached step 162,196. A 20-area continuation advanced to step 170,000, then stopped because the Editor became unresponsive during visible training. The 2,000,000-step target was not completed.
- Unity inference loaded the 50,870-step batched ONNX model and ran 20 seeds for each of PathCorridor and RoomsAndCorridors. Neither layout had a successful episode; all recorded outcomes were StepLimit.
- The Mac installation and training workflow has not been verified. Treat this as an experimental implementation, not a validated research result. Keep it separate from the built-in LSTM path so either can be disabled without changing the landmark observation design.
