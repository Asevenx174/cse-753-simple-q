# Investigating Value Overestimation in DQN and Double DQN

## Project objective

This project investigates value overestimation in Deep Q-Networks (DQN)
and evaluates whether Double DQN reduces this bias and improves learning
stability under controlled experimental conditions.

The implementation uses Python, PyTorch, Tianshou, and Gymnasium.

## Research questions

1. Why does maximizing noisy action-value estimates produce positive bias?
2. Does Double DQN reduce learned value-estimation errors and inappropriate
   risky choices in a controlled environment?
3. How do DQN and Double DQN compare in performance and learning stability
   on a standard control benchmark?
4. How do action count, target-network update frequency, and exploration
   affect the observed differences?

## Environments

### CartPole-v1

CartPole is used only to validate that the shared training and evaluation
pipeline can learn a meaningful policy.

### NoisyMax

NoisyMax is a controlled environment with analytically known action values.
It is used to directly measure value-estimation error and risky behavior.

### Acrobot-v1

Acrobot is the main standard benchmark for comparing the performance and
learning stability of DQN and Double DQN.

## Planned experiments

1. Verify the runtime and environment interaction.
2. Reproduce synthetic maximization bias using action counts
   2, 8, 32, 128, and 256.
3. Verify DQN and Double DQN targets manually and against Tianshou.
4. Validate the shared pipeline on CartPole.
5. Train both algorithms on NoisyMax using action counts 2, 10, and 50.
6. Compare both algorithms on Acrobot using three random seeds.
7. Compare target-copy intervals 320 and 1000 on NoisyMax.
8. Compare exploration rates 0.10 and 0.20 on NoisyMax.

## Fair-comparison principle

DQN and Double DQN will use the same network architecture, optimizer,
training budget, random seeds, evaluation schedule, and environment settings.

The intended algorithmic difference is the target calculation:

- DQN uses the target network for both action selection and evaluation.
- Double DQN uses the online network for selection and the target network
  for evaluation.

Experiments will change one selected factor at a time while keeping the
remaining settings fixed.

## Evaluation

The project will examine:

- Episodic return
- TD loss
- Value-estimation error
- Risky-action behavior
- Action coverage
- Learning speed
- Variation across random seeds

Reduced estimation bias does not necessarily guarantee higher episodic
return. Performance and value accuracy will therefore be interpreted
separately.