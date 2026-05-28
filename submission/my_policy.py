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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        return self.policy_head(self.trunk(obs_tensor)).argmax().item()

class MyPolicy:
    def __init__(self):
        self.agent = ActorCritic(obs_size=36, n_actions=5).to(device)
        try:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.agent.load_state_dict(checkpoint['model'], strict=True)
            self.agent.eval()
        except FileNotFoundError:
            print("⚠️ Aucun checkpoint trouvé, démarrage avec poids aléatoires")

    def act(self, obs, env=None):
        return self.agent.act(obs)
