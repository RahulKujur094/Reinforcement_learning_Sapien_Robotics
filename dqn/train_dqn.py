import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from utils.custom_cartpole import make_high_force_cartpole


class DQNAgent:
    """DQN Agent for CartPole-v1."""
    
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


def train_dqn(episodes=500, max_steps=500, save_path=None, force_magnitude=15.0):
    """
    Train DQN agent on CartPole-v1 with increased cart movement force.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_path: Path to save training plot
        force_magnitude: Force applied to cart (default: 15.0, original: 10.0)
                        Higher values = faster/more aggressive cart movement
    """
    # Set default save path
    if save_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(script_dir, "results", "dqn_rewards.png")
    
    # Create results directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use custom environment with increased force
    env = make_high_force_cartpole(force_magnitude=force_magnitude)
    agent = DQNAgent()
    
    episode_rewards = []
    moving_avg_rewards = []
    moving_avg_window = 100
    reward_window = deque(maxlen=moving_avg_window)
    
    print("Training DQN agent...")
    print(f"Device: {agent.device}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        reward_window.append(total_reward)
        moving_avg = np.mean(reward_window) if len(reward_window) == moving_avg_window else np.mean(episode_rewards)
        moving_avg_rewards.append(moving_avg)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Avg Reward (last {moving_avg_window}): {moving_avg:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(moving_avg_rewards, color='red', linewidth=2, label=f'Moving Average ({moving_avg_window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Progress - CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nResults saved to {save_path}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(save_path), "dqn_model.pth")
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode_rewards': episode_rewards
    }, model_dir)
    print(f"Model saved to {model_dir}")
    
    # Print final statistics
    final_avg = np.mean(episode_rewards[-100:])
    final_std = np.std(episode_rewards[-100:])
    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Average Reward: {final_avg:.2f} ± {final_std:.2f}")
    print(f"  Max Reward: {max(episode_rewards[-100:]):.1f}")
    print(f"  Min Reward: {min(episode_rewards[-100:]):.1f}")
    
    return agent, episode_rewards, moving_avg_rewards


if __name__ == "__main__":
    train_dqn(episodes=500)

