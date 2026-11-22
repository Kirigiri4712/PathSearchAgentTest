from typing import Any, Dict, Type, Union

from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.model_saver.model_saver import BaseModelSaver

import mlagents.trainers.ppo.optimizer_torch as ppo_optimizer_module

from .network import GTrXLSimpleActor, GTrXLValueNetwork
from .model_saver import GTrXLModelSaver


class GTrXLPPOTrainer(PPOTrainer):
    @staticmethod
    def get_trainer_name() -> str:
        return "gtrxl_ppo"

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        if self.shared_critic:
            raise ValueError("gtrxl_ppo currently requires shared_critic: false.")
        return TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            GTrXLSimpleActor,
            {"conditional_sigma": False, "tanh_squash": False},
        )

    def create_optimizer(self) -> TorchOptimizer:
        if self.shared_critic:
            raise ValueError("gtrxl_ppo currently requires shared_critic: false.")
        original_value_network = ppo_optimizer_module.ValueNetwork
        ppo_optimizer_module.ValueNetwork = GTrXLValueNetwork
        try:
            return TorchPPOOptimizer(self.policy, self.trainer_settings)
        finally:
            ppo_optimizer_module.ValueNetwork = original_value_network

    @staticmethod
    def create_model_saver(
        trainer_settings: TrainerSettings, model_path: str, load: bool
    ) -> BaseModelSaver:
        return GTrXLModelSaver(trainer_settings, model_path, load)
