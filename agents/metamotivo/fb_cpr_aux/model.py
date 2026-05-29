# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses

import torch

from .. import config_from_dict
from ..nn_models import RewardNormalizerConfig, build_forward
from ..fb_cpr.model import ArchiConfig as FBcprArchiConfig
from ..fb_cpr.model import Config as FBcprModelConfig
from ..fb_cpr.model import CriticArchiConfig
from ..fb_cpr.model import FBcprModel


@dataclasses.dataclass
class ArchiConfig(FBcprArchiConfig):
    aux_critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)


@dataclasses.dataclass
class Config(FBcprModelConfig):
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    norm_aux_reward: RewardNormalizerConfig = dataclasses.field(
        default_factory=lambda: RewardNormalizerConfig(translate=False, scale=True)
    )


class FBcprAuxModel(FBcprModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config_from_dict(kwargs, Config)
        self._aux_critic = build_forward(
            self.cfg.obs_dim,
            self.cfg.archi.z_dim,
            self.cfg.action_dim,
            self.cfg.archi.aux_critic,
            output_dim=1,
        )
        self._aux_reward_normalizer = self.cfg.norm_aux_reward.build()

        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_aux_critic = copy.deepcopy(self._aux_critic)

    @torch.no_grad()
    def aux_critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._aux_critic(self._normalize(obs), z, action)
