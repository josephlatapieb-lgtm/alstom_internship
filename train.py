import os, torch, torch.nn as nn, torch.optim as optim, sys
sys.path.append(os.getcwd())

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from submission.my_observation_builder import MyObservationBuilder
from submission.my_policy import MyPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = RailEnv(width=100, height=100,
              rail_generator=sparse_rail_generator(max_num_cities=2, seed=42),
              line_generator=sparse_line_generator(),
              number_of_agents=3,
              obs_builder_object=MyObservationBuilder())

policy = MyPolicy()
optimizer = optim.Adam(policy.agent.parameters(), lr=1e-5)
print("🤖 Entraînement 100x100 avec 3 agents lancé.")

for episode in range(1, 50001):
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    score_total = 0
    episode_loss = 0
    step_count = 0

    for step in range(100):
        actions = {}

        for h in env.get_agent_handles():
            if obs is not None and isinstance(obs, (list, dict)) and h in obs and obs[h] is not None:
                obs_tensor = torch.tensor(obs[h], dtype=torch.float32).unsqueeze(0).to(device)
                policy.agent.train()
                with torch.no_grad():
                    logits = policy.agent.policy_head(policy.agent.trunk(obs_tensor))
                actions[h] = logits.argmax().item()
            else:
                actions[h] = 0

        next_obs, rewards, dones, info = env.step(actions)

        for h in env.get_agent_handles():
            if obs is not None and isinstance(obs, (list, dict)) and h in obs and obs[h] is not None:
                obs_tensor = torch.tensor(obs[h], dtype=torch.float32).unsqueeze(0).to(device)
                reward = torch.tensor(float(rewards[h]), dtype=torch.float32, device=device)

                trunk_out = policy.agent.trunk(obs_tensor)
                logits = policy.agent.policy_head(trunk_out)
                value = policy.agent.value_head(trunk_out).squeeze()

                action_log_prob = torch.nn.functional.log_softmax(logits, dim=1)[0, actions[h]]
                advantage = reward - value.detach()

                actor_loss = -action_log_prob * advantage
                critic_loss = (advantage) ** 2

                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.agent.parameters(), 1.0)
                optimizer.step()

                episode_loss += loss.item()
                step_count += 1

        score_total += sum(rewards.values())
        obs = next_obs
        if dones['__all__']: break

    avg_loss = episode_loss / max(step_count, 1)
    if episode % 100 == 0:
        print(f"Épisode {episode} - Score : {score_total:.2f} - Loss : {avg_loss:.4f}")

    if episode % 1000 == 0:
        checkpoint_data = {'model': policy.agent.state_dict()}
        torch.save(checkpoint_data, "./submission/checkpoint.pt")
        print(f"💾 Checkpoint mis à jour à l'épisode {episode}")

print("✨ Entraînement terminé !")
