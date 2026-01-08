import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for CartPole-v1.
    Input: 4-dimensional state vector
    Output: Q-values for 2 actions
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

