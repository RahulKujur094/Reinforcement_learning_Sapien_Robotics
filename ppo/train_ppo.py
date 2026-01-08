import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_cartpole import make_high_force_cartpole


def train_ppo(total_timesteps=100000, save_path=None, force_magnitude=15.0):
    """
    Train PPO agent on CartPole-v1 using Stable-Baselines3 with increased cart movement force.
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save training plot
        force_magnitude: Force applied to cart (default: 15.0, original: 10.0)
                        Higher values = faster/more aggressive cart movement
    """
    # Set default paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_path is None:
        save_path = os.path.join(script_dir, "results", "ppo_rewards.png")
    
    results_dir = os.path.join(script_dir, "results")
    log_dir = os.path.join(results_dir, "ppo_logs")
    model_path = os.path.join(results_dir, "ppo_cartpole")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Use custom environment with increased force
    env = make_high_force_cartpole(force_magnitude=force_magnitude)
    
    # Wrap environment with Monitor to track rewards
    env = Monitor(env, log_dir)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None  # Disable tensorboard logging (optional)
    )
    
    print("Training PPO agent...")
    print(f"Total timesteps: {total_timesteps}")
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save the model
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Load training data from Monitor
    episode_rewards = []
    episode_lengths = []
    
    if os.path.exists(os.path.join(log_dir, "monitor.csv")):
        import pandas as pd
        df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
        episode_rewards = df['r'].values
        episode_lengths = df['l'].values
    
    # If monitor data not available, evaluate manually
    if len(episode_rewards) == 0:
        print("Evaluating agent performance...")
        eval_env = gym.make("CartPole-v1")
        for _ in range(100):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)
        eval_env.close()
        episode_rewards = np.array(episode_rewards)
    
    # Calculate moving average
    moving_avg_window = 100
    moving_avg_rewards = []
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - moving_avg_window + 1)
        moving_avg_rewards.append(np.mean(episode_rewards[start_idx:i+1]))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.3, color='green', label='Episode Reward')
    plt.plot(moving_avg_rewards, color='red', linewidth=2, label=f'Moving Average ({moving_avg_window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress - CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nResults saved to {save_path}")
    
    # Print final statistics
    final_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    final_avg = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    print(f"\nFinal Statistics (last {len(final_rewards)} episodes):")
    print(f"  Average Reward: {final_avg:.2f} ± {final_std:.2f}")
    print(f"  Max Reward: {max(final_rewards):.1f}")
    print(f"  Min Reward: {min(final_rewards):.1f}")
    
    env.close()
    return model, episode_rewards, moving_avg_rewards


if __name__ == "__main__":
    train_ppo(total_timesteps=100000)

