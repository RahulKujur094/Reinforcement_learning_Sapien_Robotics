# DQN 



https://github.com/user-attachments/assets/f20d2a03-c1f3-41e0-afbf-ccf86006e6f8




# PPO


https://github.com/user-attachments/assets/3af7858f-52dd-4db2-a34d-ab3043e0adac


# CartPole-v1 

Reinforcement Learning Project

This project implements and compares two reinforcement learning algorithms (DQN and PPO) on the CartPole-v1 environment from OpenAI Gym.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Results](#results)
- [Hyperparameters](#hyperparameters)
- [Comparison](#comparison)

## 🎯 Overview

The goal is to train RL agents to balance a pole on a cart by applying forces to the cart. The environment provides:
- **State Space**: 4D continuous (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: Discrete (0: push left, 1: push right)
- **Reward**: +1 for each step the pole remains upright
- **Success**: Average reward ≥ 195 over 100 consecutive episodes

## 📁 Project Structure

```
cartpole_rl/
│── dqn/
│   ├── train_dqn.py          # DQN training script
│   ├── model.py               # DQN neural network architecture
│   ├── replay_buffer.py      # Experience replay buffer
│
│── ppo/
│   ├── train_ppo.py          # PPO training script
│   ├── policy.py             # PPO policy documentation
│
│── utils/
│   ├── plot_results.py       # Visualization and comparison utilities
│   ├── record_video.py       # Video recording utilities
│
│── results/
│   ├── dqn_rewards.png       # DQN training curve
│   ├── ppo_rewards.png       # PPO training curve
│   ├── comparison.png        # Side-by-side comparison
│
│── videos/
│   ├── dqn/                  # DQN agent videos
│   ├── ppo/                  # PPO agent videos
│
│── requirements.txt
│── README.md
```

## 🚀 Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Train Both Agents (Recommended)

```bash
cd cartpole_rl
python train_all.py
```

This will:
- Train both DQN and PPO agents sequentially
- Generate individual training curves
- Create a comparison plot
- Save all results to `results/`

### Train Individual Agents

**DQN Agent:**
```bash
cd cartpole_rl
python dqn/train_dqn.py
```

This will:
- Train the DQN agent for 500 episodes
- Save training curves to `results/dqn_rewards.png`
- Save the model to `results/dqn_model.pth`
- Print training statistics

**PPO Agent:**
```bash
cd cartpole_rl
python ppo/train_ppo.py
```

This will:
- Train the PPO agent for 100,000 timesteps
- Save training curves to `results/ppo_rewards.png`
- Save the trained model to `results/ppo_cartpole`

### Record Videos

**PPO Agent:**
```bash
python utils/record_video.py --agent ppo --episodes 3
```

**Both Agents:**
```bash
python utils/record_video.py --agent both --episodes 3
```

### Compare Results

```bash
python utils/plot_results.py
```

## 🧠 Algorithms

### 1. Deep Q-Network (DQN)

**Type**: Value-based, off-policy

**Key Components**:
- **Neural Network**: 3-layer MLP (4 → 128 → 128 → 2)
- **Experience Replay**: Stores transitions in a buffer, samples batches for training
- **Target Network**: Separate network for stable Q-value targets
- **ε-Greedy Exploration**: Balances exploration vs exploitation

**How it works**:
1. Agent interacts with environment using ε-greedy policy
2. Transitions (state, action, reward, next_state, done) stored in replay buffer
3. Sample random batches from buffer to break correlation
4. Train Q-network to minimize TD error
5. Periodically update target network

**Strengths**:
- Sample efficient (reuses past experiences)
- Stable learning with experience replay
- Good baseline for comparison

**Weaknesses**:
- Requires careful hyperparameter tuning
- Can be unstable without target network
- Slower convergence than policy gradient methods

### 2. Proximal Policy Optimization (PPO)

**Type**: Policy-based, on-policy

**Key Components**:
- **Actor-Critic Architecture**: Separate networks for policy and value function
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates

**How it works**:
1. Collect trajectories by following current policy
2. Compute advantages using GAE
3. Update policy using clipped objective (prevents destructive updates)
4. Update value function to reduce variance
5. Repeat with new policy

**Strengths**:
- Very stable and robust
- Fast convergence
- Less sensitive to hyperparameters
- State-of-the-art performance on many tasks

**Weaknesses**:
- On-policy (can't reuse old data)
- Requires more environment interactions

## 📊 Results

### Expected Performance

| Metric | DQN | PPO |
|--------|-----|-----|
| **Convergence Speed** | Slower (~300-400 episodes) | Faster (~100-200 episodes) |
| **Stability** | Medium (some variance) | High (very stable) |
| **Final Average Reward** | ~200-500 | ~400-500 |
| **Sample Efficiency** | Good (replay buffer) | Moderate (on-policy) |

### Training Curves

After training, you'll find:
- `results/dqn_rewards.png`: DQN learning curve
- `results/ppo_rewards.png`: PPO learning curve
- `results/comparison.png`: Side-by-side comparison (if both trained)

## ⚙️ Hyperparameters

### DQN Hyperparameters

```python
learning_rate = 1e-3
gamma = 0.99                    # Discount factor
epsilon_start = 1.0            # Initial exploration rate
epsilon_end = 0.01              # Final exploration rate
epsilon_decay = 0.995           # Decay per episode
memory_size = 10000             # Replay buffer size
batch_size = 64                 # Training batch size
target_update = 10              # Target network update frequency
hidden_dim = 128                # Hidden layer size
```

### PPO Hyperparameters (Stable-Baselines3 Defaults)

```python
learning_rate = 3e-4
n_steps = 2048                  # Steps per update
batch_size = 64                 # Minibatch size
n_epochs = 10                   # Optimization epochs per update
gamma = 0.99                    # Discount factor
gae_lambda = 0.95               # GAE lambda
clip_range = 0.2                # PPO clip range
ent_coef = 0.01                 # Entropy coefficient
```

## 🔍 Comparison

### Learning Characteristics

**DQN**:
- Shows gradual improvement with some variance
- Benefits from experience replay
- May require more episodes to converge

**PPO**:
- Shows rapid, stable improvement
- More consistent performance
- Typically reaches optimal performance faster

### When to Use Each

**Use DQN when**:
- You want to understand value-based RL
- Sample efficiency is critical
- You have limited compute for training

**Use PPO when**:
- You want state-of-the-art performance
- Stability is important
- You have sufficient compute for on-policy learning

## 📸 Screenshots & Videos

After training and recording:
- Check `videos/dqn/` for DQN agent demonstrations
- Check `videos/ppo/` for PPO agent demonstrations
- Training plots are saved in `results/`

## 🔧 Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: CUDA out of memory
- **Solution**: DQN will automatically use CPU if CUDA unavailable

**Issue**: Videos not recording
- **Solution**: Ensure `gymnasium[classic_control]` is installed and render mode is set correctly

## 📚 References

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Human-level control through deep reinforcement learning
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization Algorithms
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📝 License

This project is for educational purposes.

---

**Happy Training! 🎉**

