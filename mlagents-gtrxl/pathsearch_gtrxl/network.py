"""GTrXL-style recurrent actor and critic for ML-Agents 1.1.0.

The network uses per-layer segment memory, relative positional attention, and
GRU-style gated residuals. It intentionally uses the existing ML-Agents
observation encoders so BufferSensor observations keep their native attention
encoding.
"""

from typing import Dict, List, Optional, Tuple

from mlagents.torch_utils import nn, torch
from mlagents_envs.base_env import ActionSpec, ObservationSpec

from mlagents.trainers.settings import ConditioningType, NetworkSettings
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.networks import (
    ObservationEncoder,
    SimpleActor,
    ValueNetwork,
)
from mlagents.trainers.torch_entities.layers import LinearEncoder
from mlagents.trainers.torch_entities.conditioning import ConditionalEncoder
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch_entities.model_serialization import exporting_to_onnx


class GatedResidual(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.reset_base = nn.Linear(hidden_size, hidden_size)
        self.reset_update = nn.Linear(hidden_size, hidden_size)
        self.candidate_base = nn.Linear(hidden_size, hidden_size)
        self.candidate_update = nn.Linear(hidden_size, hidden_size)
        self.gate_base = nn.Linear(hidden_size, hidden_size)
        self.gate_update = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.gate_base.bias, 0.0)
        nn.init.constant_(self.gate_update.bias, -2.0)

    def forward(self, base: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        reset = torch.sigmoid(self.reset_base(base) + self.reset_update(update))
        candidate = torch.tanh(
            self.candidate_base(reset * base) + self.candidate_update(update)
        )
        gate = torch.sigmoid(self.gate_base(base) + self.gate_update(update))
        return (1.0 - gate) * base + gate * candidate


class GTrXLBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context_length: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("GTrXL hidden_units must be divisible by attention_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.context_length = context_length

        self.attention_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.relative_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, self.head_size))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, self.head_size))
        self.attention_gate = GatedResidual(hidden_size)

        self.feedforward_norm = nn.LayerNorm(hidden_size)
        self.feedforward_in = nn.Linear(hidden_size, hidden_size * 4)
        self.feedforward_out = nn.Linear(hidden_size * 4, hidden_size)
        self.feedforward_gate = GatedResidual(hidden_size)

    def _relative_positions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(
            self.context_length, -1, -1, device=device, dtype=dtype
        ).unsqueeze(1)
        frequencies = torch.pow(
            torch.tensor(10000.0, device=device, dtype=dtype),
            -torch.arange(0, self.hidden_size, 2, device=device, dtype=dtype)
            / self.hidden_size,
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.stack(
            (torch.sin(angles), torch.cos(angles)), dim=-1
        ).reshape(
            self.context_length + 1, self.hidden_size
        )
        return self.relative_projection(encoding).reshape(
            self.context_length + 1, self.num_heads, self.head_size
        )

    def forward(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        memory_count: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current.shape[0]
        source = torch.cat((memory, current.unsqueeze(1)), dim=1)
        normalized = self.attention_norm(source)

        query = self.query(normalized[:, -1, :]).reshape(
            batch_size, self.num_heads, self.head_size
        )
        key = self.key(normalized).reshape(
            batch_size, self.context_length + 1, self.num_heads, self.head_size
        ).transpose(1, 2)
        value = self.value(normalized).reshape(
            batch_size, self.context_length + 1, self.num_heads, self.head_size
        ).transpose(1, 2)
        relative = self._relative_positions(current.device, current.dtype)

        query_with_content_bias = query + self.content_bias.unsqueeze(0)
        query_with_position_bias = query + self.position_bias.unsqueeze(0)
        content_scores = torch.matmul(
            query_with_content_bias.unsqueeze(2), key.transpose(-1, -2)
        )
        relative_keys = relative.permute(1, 0, 2).unsqueeze(0)
        position_scores = torch.matmul(
            query_with_position_bias.unsqueeze(2), relative_keys.transpose(-1, -2)
        )
        scores = (content_scores + position_scores) / (self.head_size ** 0.5)

        memory_positions = torch.arange(
            self.context_length, device=current.device
        ).unsqueeze(0)
        valid_memory = memory_positions >= (
            self.context_length - memory_count.to(torch.long).unsqueeze(1)
        )
        valid_positions = torch.cat(
            (
                valid_memory,
                torch.ones(batch_size, 1, dtype=torch.bool, device=current.device),
            ),
            dim=1,
        ).unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(~valid_positions, -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value).squeeze(2).reshape(
            batch_size, self.hidden_size
        )
        attended = self.attention_output(attended)
        hidden = self.attention_gate(current, attended)

        feedforward = self.feedforward_norm(hidden)
        feedforward = torch.relu(self.feedforward_in(feedforward))
        feedforward = self.feedforward_out(feedforward)
        return self.feedforward_gate(hidden, feedforward)


class GTrXLNetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if network_settings.memory is None:
            raise ValueError("The GTrXL trainer requires network_settings.memory.")

        self.normalize = network_settings.normalize
        self.h_size = network_settings.hidden_units
        self.num_layers = network_settings.num_layers
        self.m_size = network_settings.memory.memory_size
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        total_size = self.observation_encoder.total_enc_size + encoded_act_size
        if (
            self.observation_encoder.total_goal_enc_size > 0
            and network_settings.goal_conditioning_type == ConditioningType.HYPER
        ):
            self.input_encoder = ConditionalEncoder(
                total_size,
                self.observation_encoder.total_goal_enc_size,
                self.h_size,
                network_settings.num_layers,
                1,
            )
        else:
            self.input_encoder = LinearEncoder(
                total_size, network_settings.num_layers, self.h_size
            )

        per_layer_memory = self.m_size // self.num_layers
        if self.m_size % self.num_layers != 0 or (per_layer_memory - 1) % self.h_size != 0:
            raise ValueError(
                "GTrXL memory_size must equal num_layers * (context_length * hidden_units + 1)."
            )
        self.context_length = (per_layer_memory - 1) // self.h_size
        if self.context_length < 2:
            raise ValueError("GTrXL context_length must be at least two prior steps.")

        self.layers = nn.ModuleList(
            GTrXLBlock(self.h_size, attention_heads, self.context_length)
            for _ in range(self.num_layers)
        )

    @property
    def memory_size(self) -> int:
        return self.m_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "GTrXLNetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    def _unpack_memory(
        self, memory_state: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        slot_size = self.context_length * self.h_size
        per_layer_size = slot_size + 1
        layer_memories = []
        layer_counts = []
        for layer_index in range(self.num_layers):
            start = layer_index * per_layer_size
            layer_memories.append(
                memory_state[:, start : start + slot_size].reshape(
                    -1, self.context_length, self.h_size
                )
            )
            layer_counts.append(memory_state[:, start + slot_size])
        return layer_memories, layer_counts

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.observation_encoder(inputs)
        if actions is not None:
            encoded = torch.cat((encoded, actions), dim=1)
        encoded = self.input_encoder(encoded)
        batch_size = encoded.shape[0] // sequence_length
        encoded = encoded.reshape(batch_size, sequence_length, self.h_size)

        if memories is None:
            memory_state = torch.zeros(
                batch_size, self.m_size, dtype=encoded.dtype, device=encoded.device
            )
        else:
            if memories.dim() == 3:
                # Unity's recurrent_in tensor is [batch, 1, memory_size]. The
                # trainer's recurrent sequence convention is [1, batch,
                # memory_size]. A batch-size-one export made these layouts
                # indistinguishable and hid the axis error.
                memory_state = (
                    memories[:, 0, :]
                    if exporting_to_onnx.is_exporting()
                    else memories[0]
                )
            else:
                memory_state = memories
        layer_memories, layer_counts = self._unpack_memory(memory_state)

        output_sequence = []
        for step_index in range(sequence_length):
            hidden = encoded[:, step_index, :]
            next_memories = []
            next_counts = []
            for layer_index, block in enumerate(self.layers):
                hidden = block(
                    hidden,
                    layer_memories[layer_index],
                    layer_counts[layer_index],
                )
                next_memory = torch.cat(
                    (layer_memories[layer_index][:, 1:, :], hidden.unsqueeze(1)),
                    dim=1,
                )
                next_memories.append(next_memory)
                next_counts.append(torch.clamp(layer_counts[layer_index] + 1.0, max=self.context_length))
            layer_memories = next_memories
            layer_counts = next_counts
            output_sequence.append(hidden)

        packed_memory = []
        for memory, count in zip(layer_memories, layer_counts):
            packed_memory.extend((memory.reshape(batch_size, -1), count.unsqueeze(1)))
        memory_out = torch.cat(packed_memory, dim=1)
        if exporting_to_onnx.is_exporting():
            # Unity's recurrent input generator and output applier both use
            # [batch, 1, memory_size]. Keep the exported policy batch-first.
            memory_out = memory_out.unsqueeze(1)
        else:
            memory_out = memory_out.unsqueeze(0)
        return torch.stack(output_sequence, dim=1).reshape(-1, self.h_size), memory_out


class GTrXLSimpleActor(SimpleActor):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.action_spec = action_spec
        self.version_number = nn.Parameter(torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False)
        self.is_continuous_int_deprecated = nn.Parameter(
            torch.Tensor([int(action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = nn.Parameter(
            torch.Tensor([int(action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = nn.Parameter(
            torch.Tensor([action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = nn.Parameter(
            torch.Tensor([action_spec.continuous_size + sum(action_spec.discrete_branches)]),
            requires_grad=False,
        )
        self.network_body = GTrXLNetworkBody(observation_specs, network_settings)
        self.encoding_size = network_settings.hidden_units
        self.memory_size_vector = nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )


class GTrXLValueNetwork(ValueNetwork):
    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ) -> None:
        nn.Module.__init__(self)
        self.network_body = GTrXLNetworkBody(
            observation_specs,
            network_settings,
            encoded_act_size=encoded_act_size,
        )
        self.value_heads = ValueHeads(
            stream_names, network_settings.hidden_units, outputs_per_stream
        )
