import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder

class MyObservationBuilder(ObservationBuilder):
    def reset(self):
        return True

    def get(self, handle):
        agent = self.env.agents[handle]
        obs = np.zeros(36, dtype=np.float32)

        # Position normalisée (2 features)
        if agent.position is not None:
            obs[0] = agent.position[0] / max(1, self.env.width)
            obs[1] = agent.position[1] / max(1, self.env.height)

        # Direction (4 features - one-hot)
        if agent.direction is not None and 0 <= agent.direction < 4:
            obs[2 + agent.direction] = 1.0

        # Target normalisé (2 features)
        if agent.target is not None:
            obs[6] = agent.target[0] / max(1, self.env.width)
            obs[7] = agent.target[1] / max(1, self.env.height)

        # Distance à la cible (1 feature)
        if agent.position is not None and agent.target is not None:
            dist = np.sqrt((agent.position[0] - agent.target[0])**2 +
                          (agent.position[1] - agent.target[1])**2)
            obs[8] = min(dist / (self.env.width + self.env.height), 1.0)

        # Infos du chemin si disponible (14 features)
        if hasattr(agent, 'path') and agent.path and len(agent.path) > 0:
            for i in range(min(7, len(agent.path))):
                try:
                    next_pos = agent.path[i]
                    obs[9 + i*2] = next_pos[0] / max(1, self.env.width)
                    obs[10 + i*2] = next_pos[1] / max(1, self.env.height)
                except (TypeError, IndexError):
                    pass

        return obs
