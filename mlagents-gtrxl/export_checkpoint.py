"""Export a saved GTrXL .pt checkpoint without resuming or advancing training.

Example:
    .venv-mlagents/Scripts/python.exe mlagents-gtrxl/export_checkpoint.py \
        --checkpoint results/pathsearch_gtrxl_short_r2_20261121/ExplorerAgent/ExplorerAgent-50870.pt \
        --schema Assets/Models/ExplorerAgent-GTrXL-50870.onnx \
        --output Assets/Models/ExplorerAgent-GTrXL-50870-batched.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import onnx
import yaml
from mlagents.torch_utils import torch
from mlagents.trainers.settings import ConditioningType, EncoderType, NetworkSettings
from mlagents_envs.base_env import (
    ActionSpec,
    BehaviorSpec,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
)

from pathsearch_gtrxl.model_saver import GTrXLModelSerializer
from pathsearch_gtrxl.network import GTrXLSimpleActor


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "results/pathsearch_gtrxl_short_r2_20261121/ExplorerAgent/ExplorerAgent-50870.pt"
)
DEFAULT_SCHEMA = REPOSITORY_ROOT / "Assets/Models/ExplorerAgent-GTrXL-50870.onnx"
DEFAULT_CONFIG = REPOSITORY_ROOT / "Assets/config/trainer_gtrxl.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "Assets/Models/ExplorerAgent-GTrXL-50870-batched.onnx"


def _absolute_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else REPOSITORY_ROOT / result


def _observation_specs(schema_path: Path) -> List[ObservationSpec]:
    schema = onnx.load(str(schema_path))
    observation_inputs = []
    for value in schema.graph.input:
        if not value.name.startswith("obs_"):
            continue
        index_text = value.name.removeprefix("obs_")
        if not index_text.isdigit():
            raise ValueError(f"Unexpected observation input name: {value.name}")
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) < 2:
            raise ValueError(f"Unexpected observation rank for {value.name}.")
        shape = []
        for dimension in dimensions[1:]:
            if dimension.dim_value <= 0:
                raise ValueError(
                    f"Observation {value.name} has a non-static feature dimension."
                )
            shape.append(dimension.dim_value)
        observation_inputs.append((int(index_text), value.name, tuple(shape)))

    observation_inputs.sort(key=lambda item: item[0])
    if not observation_inputs:
        raise ValueError(f"No obs_<index> inputs found in {schema_path}.")

    specs = []
    for _, name, shape in observation_inputs:
        if len(shape) == 2:
            properties = (
                DimensionProperty.VARIABLE_SIZE,
                DimensionProperty.NONE,
            )
        else:
            properties = tuple(DimensionProperty.NONE for _ in shape)
        specs.append(
            ObservationSpec(
                shape=shape,
                dimension_property=properties,
                observation_type=ObservationType.DEFAULT,
                name=name,
            )
        )
    return specs


def _network_settings(config_path: Path) -> NetworkSettings:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    behavior = config["behaviors"]["ExplorerAgent"]
    settings = behavior["network_settings"]
    memory = settings["memory"]
    return NetworkSettings(
        normalize=bool(settings.get("normalize", False)),
        hidden_units=int(settings.get("hidden_units", 128)),
        num_layers=int(settings.get("num_layers", 2)),
        vis_encode_type=EncoderType(settings.get("vis_encode_type", "simple")),
        memory=NetworkSettings.MemorySettings(
            sequence_length=int(memory["sequence_length"]),
            memory_size=int(memory["memory_size"]),
        ),
        goal_conditioning_type=ConditioningType(
            settings.get("goal_conditioning_type", "hyper")
        ),
        deterministic=bool(settings.get("deterministic", False)),
    )


def export_checkpoint(
    checkpoint_path: Path,
    schema_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"ONNX schema model not found: {schema_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Trainer config not found: {config_path}")

    # This project's behavior uses two continuous actions and one binary
    # discrete branch; keep these aligned with Player.prefab.
    action_spec = ActionSpec.create_hybrid(2, (2,))
    behavior_spec = BehaviorSpec(
        _observation_specs(schema_path),
        action_spec,
    )
    actor = GTrXLSimpleActor(
        observation_specs=behavior_spec.observation_specs,
        network_settings=_network_settings(config_path),
        action_spec=action_spec,
        conditional_sigma=False,
        tanh_squash=False,
    )

    payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    policy_state = payload.get("Policy")
    if not isinstance(policy_state, dict):
        raise ValueError("Checkpoint does not contain the expected 'Policy' state.")
    actor.load_state_dict(policy_state, strict=True)
    actor.eval()

    policy_view = SimpleNamespace(
        actor=actor,
        behavior_spec=behavior_spec,
        export_memory_size=actor.memory_size,
    )
    serializer = GTrXLModelSerializer(policy_view)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_stem = output_path.with_name(output_path.stem + ".tmp-export")
    temporary_model = Path(str(temporary_stem) + ".onnx")
    if temporary_model.exists():
        temporary_model.unlink()

    try:
        serializer.export_policy_model(str(temporary_stem))
        exported = onnx.load(str(temporary_model))
        onnx.checker.check_model(exported)
        onnx.save(exported, str(temporary_model))
        os.replace(temporary_model, output_path)
    finally:
        if temporary_model.exists():
            temporary_model.unlink()

    print(f"Exported corrected ONNX: {output_path}")
    print(f"Source checkpoint: {checkpoint_path}")
    print("The checkpoint weights were loaded as-is; no training was started.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    export_checkpoint(
        _absolute_path(args.checkpoint),
        _absolute_path(args.schema),
        _absolute_path(args.config),
        _absolute_path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
