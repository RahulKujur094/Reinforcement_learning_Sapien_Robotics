import gymnasium as gym
import os
import torch
from gymnasium.wrappers import RecordVideo
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.model import DQN
from dqn.train_dqn import DQNAgent
from utils.custom_cartpole import make_high_force_cartpole


def record_dqn_video(agent, num_episodes=3, video_dir=None, force_magnitude=15.0):
    """
    Record video of DQN agent playing CartPole-v1 with increased cart movement force.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of episodes to record
        video_dir: Directory to save videos
        force_magnitude: Force applied to cart (default: 15.0, original: 10.0)
    """
    if video_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_dir = os.path.join(script_dir, "videos", "dqn")
    os.makedirs(video_dir, exist_ok=True)
    
    # Use custom environment with increased force
    env = make_high_force_cartpole(force_magnitude=force_magnitude, render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    
    print(f"Recording {num_episodes} episodes for DQN agent...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f}")
    
    env.close()
    print(f"Videos saved to {video_dir}")


def record_ppo_video(model_path=None, num_episodes=3, video_dir=None, force_magnitude=15.0):
    """
    Record video of PPO agent playing CartPole-v1 with increased cart movement force.
    
    Args:
        model_path: Path to saved PPO model
        num_episodes: Number of episodes to record
        video_dir: Directory to save videos
        force_magnitude: Force applied to cart (default: 15.0, original: 10.0)
    """
    from stable_baselines3 import PPO
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(script_dir, "results", "ppo_cartpole")
    if video_dir is None:
        video_dir = os.path.join(script_dir, "videos", "ppo")
    
    os.makedirs(video_dir, exist_ok=True)
    
    # Load model first (without environment to avoid wrapping issues)
    model = PPO.load(model_path)
    
    # Create fresh environment for recording with increased force
    env = make_high_force_cartpole(force_magnitude=force_magnitude, render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    
    print(f"Recording {num_episodes} episodes for PPO agent...")
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 1000:  # Safety limit
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
            
            print(f"Episode {episode + 1}: Reward = {total_reward:.1f}")
    except Exception as e:
        print(f"Error during recording: {e}")
        raise
    finally:
        # Ensure environment is properly closed to save videos
        try:
            env.close()
        except:
            pass
    
    print(f"Videos saved to {video_dir}")


def load_dqn_agent(model_path=None):
    """
    Load a trained DQN agent from saved checkpoint.
    
    Args:
        model_path: Path to saved DQN model (.pth file)
    
    Returns:
        DQNAgent: Loaded agent
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(script_dir, "results", "dqn_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DQN model not found at {model_path}. Please train the agent first.")
    
    # Create agent
    agent = DQNAgent()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.target_network.load_state_dict(checkpoint['target_network'])
    agent.epsilon = 0.0  # No exploration during evaluation
    
    return agent


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Record videos of trained agents")
    parser.add_argument("--agent", type=str, choices=["dqn", "ppo", "both"], default="both",
                       help="Which agent to record")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to record")
    
    args = parser.parse_args()
    
    if args.agent in ["dqn", "both"]:
        try:
            agent = load_dqn_agent()
            record_dqn_video(agent, num_episodes=args.episodes)
        except Exception as e:
            print(f"Error recording DQN: {e}")
    
    if args.agent in ["ppo", "both"]:
        try:
            record_ppo_video(num_episodes=args.episodes)
        except Exception as e:
            print(f"Error recording PPO: {e}")

