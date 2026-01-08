"""
Main script to train both DQN and PPO agents and generate comparison plots.
"""
import os
import sys
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn.train_dqn import train_dqn
from ppo.train_ppo import train_ppo
from utils.plot_results import plot_comparison, print_statistics


def main():
    """Train both agents and generate comparison."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    print("="*60)
    print("CartPole-v1 Reinforcement Learning Training")
    print("="*60)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Train DQN
    print("\n" + "="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    dqn_agent, dqn_rewards, dqn_ma = train_dqn(episodes=500)
    
    # Save DQN rewards
    np.save(os.path.join(results_dir, "dqn_rewards.npy"), np.array(dqn_rewards))
    
    # Train PPO
    print("\n" + "="*60)
    print("TRAINING PPO AGENT")
    print("="*60)
    ppo_model, ppo_rewards, ppo_ma = train_ppo(total_timesteps=100000)
    
    # Save PPO rewards
    np.save(os.path.join(results_dir, "ppo_rewards.npy"), np.array(ppo_rewards))
    
    # Generate comparison
    print("\n" + "="*60)
    print("GENERATING COMPARISON")
    print("="*60)
    plot_comparison(dqn_rewards, ppo_rewards)
    print_statistics(dqn_rewards, ppo_rewards)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved in {results_dir}")
    print("You can now record videos using:")
    print("  python utils/record_video.py --agent both --episodes 3")


if __name__ == "__main__":
    main()

