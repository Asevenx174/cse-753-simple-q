"""Train DQN or Double DQN on CartPole."""

import argparse
import platform
import subprocess

import gymnasium as gym
import numpy as np
import tianshou
import torch

from tianshou.algorithm.algorithm_base import (
    policy_within_training_step,
)
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from src.agents.dqn_factory import build_dqn
from src.agents.q_network import QNetwork
from src.utils.evaluation import evaluate_greedy
from src.utils.result_saving import (
    create_run_directory,
    save_checkpoint,
    save_csv,
    save_json,
)


ENVIRONMENT = "CartPole-v1"
SEED = 42
TOTAL_STEPS = 50_000
WARMUP_STEPS = 1_000
COLLECTION_STEPS = 10
BATCH_SIZE = 64
EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 10