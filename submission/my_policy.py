import torch
from reinforcement_learning.my_policy import ActorCritic

class MyPolicy:
    def __init__(self):
        self.agent = ActorCritic(obs_size=36, n_actions=5)

        checkpoint = torch.load("submission/checkpoint.pt", map_location="cpu")
        self.agent.load_state_dict(checkpoint["model"])
        self.agent.eval()

    def act(self, obs, env):
        actions = {}
        for h in env.get_agent_handles():
            if obs[h] is not None:
                actions[h] = self.agent.act(obs[h])
            else:
                actions[h] = 0
        return actions