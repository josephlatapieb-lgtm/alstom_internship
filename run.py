import os
import torch
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from reinforcement_learning.my_policy import ActorCritic
from reinforcement_learning.my_observation_builder import MyObservationBuilder

print("🤖 Initialisation de l'environnement...")
obs_builder = MyObservationBuilder()
env = RailEnv(
    width=100, height=100,
    rail_generator=sparse_rail_generator(max_num_cities=2, seed=42),
    line_generator=sparse_line_generator(), 
    number_of_agents=10,
    obs_builder_object=obs_builder
)

# Initialisation de l'agent
agent = ActorCritic(obs_size=36, n_actions=5)
checkpoint_path = "./submission/checkpoint.pt"

# --- Chargement automatique (sans confirmation) ---
if os.path.exists(checkpoint_path):
    print(f"📂 Savoir trouvé ! Chargement automatique du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent.load_state_dict(checkpoint['model'])
    print("✅ Le savoir a été chargé avec succès, poursuite de l'entraînement...")
else:
    print("🆕 Aucun checkpoint trouvé, l'IA commence son apprentissage à zéro.")

# --- BOUCLE D'ENTRAÎNEMENT ---
# On boucle jusqu'à 50 000
for episode in range(1, 50001):
    obs, info = env.reset()
    score_total = 0
    
    for step in range(100):
        # Vérification dimension (uniquement au premier pas)
        if episode == 1 and step == 0 and obs[0] is not None:
            taille_reelle = len(obs[0])
            taille_attendue = agent.obs_size + agent.n_actions
            print(f"🔍 Vérification dimension : réelle={taille_reelle}, attendue={taille_attendue}")

        # Action de chaque agent
        actions = {h: (agent.act(obs[h]) if obs[h] is not None else 0) for h in env.get_agent_handles()}
        next_obs, rewards, dones, info = env.step(actions)
        
        score_total += sum(rewards.values())
        obs = next_obs
        if dones['__all__']: break
    
    # Affichage régulier pour suivre l'évolution
    if episode % 100 == 0 or episode == 1:
        print(f"Épisode {episode}/50000 - Score total : {score_total:.2f}")
    
    # Sauvegarde automatique à chaque palier (fréquence augmentée pour sécurité)
    if episode % 1000 == 0:
        torch.save({"model": agent.state_dict()}, checkpoint_path)
        print(f"💾 Sauvegarde effectuée : épisode {episode}")

# Sauvegarde finale à la fin du script
torch.save({"model": agent.state_dict()}, checkpoint_path)
print("✨ Entraînement terminé et état final sauvegardé !")