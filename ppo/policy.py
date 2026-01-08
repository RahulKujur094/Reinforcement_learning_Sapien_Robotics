"""
PPO Policy Module

This module uses Stable-Baselines3's PPO implementation.
The policy is automatically created when instantiating PPO with "MlpPolicy".

For custom policy implementation, you can extend:
- stable_baselines3.common.policies.ActorCriticPolicy
- stable_baselines3.common.policies.MlpPolicy

The default MlpPolicy uses:
- Actor: MLP with 2 hidden layers (64 units each)
- Critic: MLP with 2 hidden layers (64 units each)
- Activation: tanh
- Action distribution: Categorical (for discrete actions)
"""

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# The PPO implementation in train_ppo.py uses the default MlpPolicy
# which is sufficient for CartPole-v1. This file documents the policy structure.

__all__ = ['PPO', 'ActorCriticPolicy']

