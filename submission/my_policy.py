import os
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
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.policy_head(self.trunk(obs_tensor))
            return logits.argmax(dim=1).item()

class MyPolicy:
    def __init__(self):
        self.agent = ActorCritic(obs_size=36, n_actions=5).to(device)
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                self.agent.load_state_dict(checkpoint['model'], strict=False)
                self.agent.eval()
            except Exception as e:
                pass

    def act(self, obs, env=None):
        return self.agent.act(obs)
