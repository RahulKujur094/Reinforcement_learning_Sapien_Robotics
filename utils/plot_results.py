import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_rewards(file_path):
    """Load episode rewards from a file."""
    if os.path.exists(file_path):
        return np.loadtxt(file_path)
    return None


def plot_comparison(dqn_rewards=None, ppo_rewards=None, save_path=None):
    """
    Compare DQN and PPO learning curves.
    
    Args:
        dqn_rewards: Array of DQN episode rewards
        ppo_rewards: Array of PPO episode rewards
        save_path: Path to save the comparison plot
    """
    if save_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(script_dir, "results", "comparison.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 8))
    
    if dqn_rewards is not None:
        window = 100
        dqn_ma = []
        for i in range(len(dqn_rewards)):
            start_idx = max(0, i - window + 1)
            dqn_ma.append(np.mean(dqn_rewards[start_idx:i+1]))
        
        plt.plot(dqn_rewards, alpha=0.2, color='blue', label='DQN Episode Reward')
        plt.plot(dqn_ma, color='blue', linewidth=2, label=f'DQN Moving Average ({window})')
    
    if ppo_rewards is not None:
        window = 100
        ppo_ma = []
        for i in range(len(ppo_rewards)):
            start_idx = max(0, i - window + 1)
            ppo_ma.append(np.mean(ppo_rewards[start_idx:i+1]))
        
        plt.plot(ppo_rewards, alpha=0.2, color='green', label='PPO Episode Reward')
        plt.plot(ppo_ma, color='green', linewidth=2, label=f'PPO Moving Average ({window})')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('DQN vs PPO - CartPole-v1 Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")


def print_statistics(dqn_rewards=None, ppo_rewards=None):
    """Print comparison statistics."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if dqn_rewards is not None:
        final_dqn = dqn_rewards[-100:] if len(dqn_rewards) >= 100 else dqn_rewards
        print(f"\nDQN (last {len(final_dqn)} episodes):")
        print(f"  Average Reward: {np.mean(final_dqn):.2f} ± {np.std(final_dqn):.2f}")
        print(f"  Max Reward: {max(final_dqn):.1f}")
        print(f"  Min Reward: {min(final_dqn):.1f}")
        print(f"  Convergence: {'Yes' if np.mean(final_dqn) > 195 else 'Partial'}")
    
    if ppo_rewards is not None:
        final_ppo = ppo_rewards[-100:] if len(ppo_rewards) >= 100 else ppo_rewards
        print(f"\nPPO (last {len(final_ppo)} episodes):")
        print(f"  Average Reward: {np.mean(final_ppo):.2f} ± {np.std(final_ppo):.2f}")
        print(f"  Max Reward: {max(final_ppo):.1f}")
        print(f"  Min Reward: {min(final_ppo):.1f}")
        print(f"  Convergence: {'Yes' if np.mean(final_ppo) > 195 else 'Partial'}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dqn_path = os.path.join(script_dir, "results", "dqn_rewards.npy")
    ppo_path = os.path.join(script_dir, "results", "ppo_rewards.npy")
    
    dqn_rewards = None
    ppo_rewards = None
    
    if os.path.exists(dqn_path):
        dqn_rewards = np.load(dqn_path)
    if os.path.exists(ppo_path):
        ppo_rewards = np.load(ppo_path)
    
    if dqn_rewards is not None or ppo_rewards is not None:
        plot_comparison(dqn_rewards, ppo_rewards)
        print_statistics(dqn_rewards, ppo_rewards)
    else:
        print("No reward files found. Please train the agents first.")

