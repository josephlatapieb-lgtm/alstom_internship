import torch
from reinforcement_learning.my_policy import ActorCritic

# 1. On crée une IA SANS checkpoint (elle sera donc totalement aléatoire)
agent_aleatoire = ActorCritic(obs_size=36, n_actions=5, checkpoint_path=None)

# 2. On affiche le poids du premier neurone pour voir à quoi il ressemble
print(f"Poids aléatoires (début) : {agent_aleatoire.trunk[0].weight[0][0].item():.4f}")

# 3. Maintenant on charge TON checkpoint
agent_entraine = ActorCritic(obs_size=36, n_actions=5, checkpoint_path="./submission/checkpoint.pt")
print(f"Poids de TON IA (après 20k épisodes) : {agent_entraine.trunk[0].weight[0][0].item():.4f}")