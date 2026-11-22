"""ONNX serializer and model saver for the project's GTrXL actor."""

from typing import Union

from mlagents.trainers.model_saver.torch_model_saver import TorchModelSaver
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.model_serialization import (
    ModelSerializer,
    TensorNames,
)


class GTrXLModelSerializer(ModelSerializer):
    """Preserve a dynamic batch axis on exported recurrent outputs."""

    def __init__(self, policy: TorchPolicy) -> None:
        super().__init__(policy)
        if policy.export_memory_size > 0:
            self.dynamic_axes[TensorNames.recurrent_output] = {0: "batch"}


class GTrXLModelSaver(TorchModelSaver):
    """Use the GTrXL-aware serializer for every saved checkpoint."""

    def register(self, module: Union[TorchPolicy, TorchOptimizer]) -> None:
        super().register(module)
        if isinstance(module, TorchPolicy) and self.policy is module:
            self.exporter = GTrXLModelSerializer(module)
