import os
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(ActorCritic, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def act(self, obs):
        with torch.no_grad():
            if isinstance(obs, list):
                obs = np.array(obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.policy_head(self.trunk(obs_tensor))
            return int(logits.argmax(dim=1).item())

class MyPolicy:
    def __init__(self):
        self.agent = ActorCritic(obs_size=36, n_actions=5).to(device)
        self.agent.eval()

        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")

        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                self.agent.load_state_dict(checkpoint['model'], strict=False)
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    def act(self, obs, env=None):
        try:
            return self.agent.act(obs)
        except Exception as e:
            print(f"Error in act(): {e}")
            return 0
