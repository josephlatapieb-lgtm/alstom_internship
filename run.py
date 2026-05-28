import os, sys
sys.path.append(os.getcwd())

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from submission.my_observation_builder import MyObservationBuilder
from submission.my_policy import MyPolicy

env = RailEnv(
    width=25, height=25,
    rail_generator=sparse_rail_generator(max_num_cities=2, seed=1),
    line_generator=sparse_line_generator(),
    number_of_agents=1,
    obs_builder_object=MyObservationBuilder()
)

policy = MyPolicy()
print("🚂 Test de la politique chargée...")

for episode in range(3):
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    done = {"__all__": False}
    total_reward = 0

    while not done["__all__"]:
        actions = {}
        for handle in env.get_agent_handles():
            if obs[handle] is not None:
                actions[handle] = policy.act(obs[handle])
            else:
                actions[handle] = 0
        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards.values())

    print(f"Épisode {episode+1} - Reward total : {total_reward:.2f}")

print("✅ Test terminé.")
