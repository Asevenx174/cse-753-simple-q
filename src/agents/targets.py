"""Reference DQN and Double DQN target calculations."""

import torch


def masked_action_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Exclude invalid actions from maximization."""

    if action_mask is None:
        return q_values

    mask = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=q_values.device,
    )

    if mask.shape != q_values.shape:
        raise ValueError(
            "action_mask must match the Q-value shape."
        )

    if not mask.any():
        raise ValueError(
            "A nonterminal state must have a valid action."
        )

    return q_values.masked_fill(mask.logical_not(), -torch.inf)


def dqn_target(
    reward: float,
    gamma: float,
    terminated: bool,
    target_q: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> float:
    """Calculate a masked DQN target."""

    if terminated:
        return reward

    valid_target_q = masked_action_values(
        target_q,
        action_mask,
    )
    next_value = valid_target_q.max().item()

    return reward + gamma * next_value


def double_dqn_target(
    reward: float,
    gamma: float,
    terminated: bool,
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> float:
    """Calculate a masked Double-DQN target."""

    if terminated:
        return reward

    valid_online_q = masked_action_values(
        online_q,
        action_mask,
    )
    selected_action = valid_online_q.argmax().item()
    next_value = target_q[selected_action].item()

    return reward + gamma * next_value