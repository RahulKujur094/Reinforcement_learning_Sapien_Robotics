"""
Custom CartPole environment with increased cart movement force.
This wrapper/modification increases the force applied to the cart for more visible movement.
"""
import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
import numpy as np


class HighForceCartPoleEnv(CartPoleEnv):
    """
    Custom CartPole environment with increased force magnitude.
    This makes the cart move faster/more aggressively in left/right directions.
    """
    
    def __init__(self, force_magnitude=15.0, **kwargs):
        """
        Initialize CartPole with custom force magnitude.
        
        Args:
            force_magnitude: Force applied to cart (default CartPole uses 10.0)
                           Higher values = faster/more aggressive cart movement
            **kwargs: Additional arguments passed to CartPoleEnv (e.g., render_mode)
        """
        super().__init__(**kwargs)
        self.force_mag = force_magnitude
    
    def step(self, action):
        """
        Override step to use custom force magnitude.
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/2005_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated=True. You "
                    "should always call 'reset()' once you receive "
                    "terminated=True -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


def make_high_force_cartpole(force_magnitude=15.0, **kwargs):
    """
    Factory function to create a CartPole environment with increased force.
    
    Args:
        force_magnitude: Force applied to cart (default: 15.0, original: 10.0)
                        Increase this value for faster cart movement
        **kwargs: Additional arguments (e.g., render_mode)
    
    Returns:
        HighForceCartPoleEnv: Custom environment instance
    """
    return HighForceCartPoleEnv(force_magnitude=force_magnitude, **kwargs)

