import torch
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from reinforcement_learning.my_policy import ActorCritic
from reinforcement_learning.my_observation_builder import MyObservationBuilder

# 1. Configuration
obs_builder = MyObservationBuilder()
env = RailEnv(width=50, height=50, 
              rail_generator=sparse_rail_generator(max_num_cities=2, seed=1), 
              line_generator=sparse_line_generator(), 
              number_of_agents=2, 
              obs_builder_object=obs_builder)

# 2. Charger ton IA entraînée
agent = ActorCritic(obs_size=36, n_actions=5, checkpoint_path="./submission/checkpoint.pt")
agent.eval() # On met l'IA en mode évaluation (pas d'apprentissage)

# 3. Lancer un épisode de test
obs, info = env.reset()
score = 0
for step in range(100):
    actions = {h: agent.act(obs[h]) if obs[h] is not None else 0 for h in env.get_agent_handles()}
    obs, rewards, dones, info = env.step(actions)
    score += sum(rewards.values())
    if dones['__all__']: break

print(f"🎯 Score de l'IA sur cet épisode de test : {score:.2f}")