"""Reference DQN and Double DQN target calculations."""

import torch


def dqn_target(
    reward: float,
    gamma: float,
    terminated: bool,
    target_q: torch.Tensor,
) -> float:
    """Calculate the DQN target for one transition."""
    if terminated:
        return reward

    # find the largest target netowrk value
    next_value = torch.max(target_q).item()
    return reward + gamma * next_value


def double_dqn_target(
    reward: float,
    gamma: float,
    terminated: bool,
    online_q: torch.Tensor,
    target_q: torch.Tensor,
) -> float:
    """Calculate the Double DQN target for one transition."""
    if terminated:
        return reward

    # selected by online network
    selected_action = torch.argmax(online_q).item()
    # evaluate by the target network
    next_value = target_q[selected_action].item()

    return reward + gamma * next_value