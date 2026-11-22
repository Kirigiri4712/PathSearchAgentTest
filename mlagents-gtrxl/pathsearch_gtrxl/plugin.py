from typing import Any, Dict, Tuple

from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from .trainer import GTrXLPPOTrainer


def get_type_and_settings() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return {"gtrxl_ppo": GTrXLPPOTrainer}, {"gtrxl_ppo": PPOSettings}
