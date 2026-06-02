import os, sys, numpy as np
sys.path.append(os.getcwd())

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from submission.my_observation_builder import MyObservationBuilder
from submission.my_policy import MyPolicy

env = RailEnv(width=100, height=100,
              rail_generator=sparse_rail_generator(max_num_cities=2, seed=42),
              line_generator=sparse_line_generator(),
              number_of_agents=3,
              obs_builder_object=MyObservationBuilder())

policy = MyPolicy()
print("[TRAINING] Entraînement 100x100 avec 3 agents lancé.")

for episode in range(1, 50001):
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    score_total = 0
    step_count = 0

    for step in range(100):
        actions = {}

        for h in env.get_agent_handles():
            actions[h] = policy.act(obs, env)

        next_obs, rewards, dones, info = env.step(actions)
        score_total += sum(rewards.values())
        obs = next_obs
        step_count += 1

        if dones['__all__']:
            break

    if episode % 100 == 0:
        print(f"Épisode {episode} - Score : {score_total:.2f}")

    if episode % 1000 == 0:
        checkpoint_data = {'score': score_total}
        np.save("./submission/checkpoint.npy", checkpoint_data)
        print(f"[CHECKPOINT] Checkpoint mis à jour à l'épisode {episode}")

print("[DONE] Entraînement terminé !")
