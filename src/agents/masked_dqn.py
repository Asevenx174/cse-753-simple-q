"""DQN implementation with masked bootstrap maximization."""

import numpy as np
import torch

from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data import Batch, ReplayBuffer


class MaskedDQN(DQN):
    """Apply next-state action masks to DQN targets."""

    def _target_q(
        self,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> torch.Tensor:
        """Calculate masked target-network action values."""

        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )

        online_result = self.policy(obs_next_batch)

        if self.use_target_network:
            target_q = self.policy(
                obs_next_batch,
                model=self.model_old,
            ).logits
        else:
            target_q = online_result.logits

        if self.is_double:
            return target_q[
                np.arange(len(online_result.act)),
                online_result.act,
            ]

        mask = getattr(obs_next_batch.obs, "mask", None)
        masked_target_q = self.policy.compute_q_value(
            target_q,
            mask,
        )

        return masked_target_q.max(dim=1).values