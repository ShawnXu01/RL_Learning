import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQNModel:
    """Wrapper around a PyTorch Network providing predict/save/load helpers."""
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = Network(self.state_size, self.action_size).to(self.device)

    def predict(self, state):
        """Accepts numpy array or torch tensor. Returns numpy array of Q-values."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                x = torch.from_numpy(state.astype(np.float32)).to(self.device)
            else:
                x = torch.tensor(state, dtype=torch.float32).to(self.device)
            q = self.model(x)
            return q.cpu().numpy()

    def save(self, path):
        # Ensure directory exists
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
