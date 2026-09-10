# Investigating Value Overestimation in DQN and Double DQN

## Project overview

This project is a step-by-step rebuild of my earlier DQN and Double DQN
project. The submitted project proposal defines the project scope, while the
submitted project update contains preliminary results that the new
implementation will attempt to reproduce and investigate more reliably.

## Research question

Does Double DQN reduce the value overestimation produced by DQN, and under
what conditions does this reduction affect learning performance?

## Project objectives

1. Demonstrate maximization bias using a controlled synthetic experiment.
2. Implement and verify the DQN and Double DQN target calculations.
3. Train DQN and Double DQN agents using Tianshou.
4. Evaluate the algorithms on CartPole, Acrobot, and a custom NoisyMax
   environment.
5. Investigate the effects of action count, target-network update frequency,
   and exploration rate.
6. Compare results using multiple random seeds and appropriate diagnostics.
7. Explain why different algorithm settings produce different results.

## Planned environments

### CartPole-v1

CartPole will be used as an initial validation environment to confirm that the
training pipeline can learn a relatively simple discrete-action control task.

### Acrobot-v1

Acrobot will be used as the principal standard benchmark for comparing DQN
and Double DQN across multiple random seeds.

### NoisyMax

A custom controlled environment will be used to measure value overestimation
against analytically known values and to study the effect of action-space
size.

## Experimental principle

Each comparison will change one experimental factor while holding the other
settings fixed. DQN and Double DQN will use matching network architectures,
training budgets, configurations, evaluation procedures, and random seeds.

The project will measure both policy performance and value-estimation
behavior. A reduction in value overestimation will not automatically be
interpreted as an improvement in episodic return.

## Reproducibility

Experiment configurations, random seeds, dependency versions, raw results,
and the corresponding Git commit will be recorded. Preliminary results from
the earlier project update will be treated as historical observations rather
than results that the new implementation must artificially reproduce.