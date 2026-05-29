import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder

class MyObservationBuilder(ObservationBuilder):
    def reset(self):
        return True

    def get(self, handle):
        agent = self.env.agents[handle]
        obs = np.zeros(36, dtype=np.float32)

        try:
            # Position normalisée
            if agent.position is not None:
                obs[0] = float(agent.position[0]) / max(1, self.env.width)
                obs[1] = float(agent.position[1]) / max(1, self.env.height)

            # Direction (one-hot)
            if agent.direction is not None and 0 <= int(agent.direction) < 4:
                obs[2 + int(agent.direction)] = 1.0

            # Target
            if agent.target is not None:
                obs[6] = float(agent.target[0]) / max(1, self.env.width)
                obs[7] = float(agent.target[1]) / max(1, self.env.height)

            # Distance to target
            if agent.position is not None and agent.target is not None:
                dist = np.sqrt((float(agent.position[0]) - float(agent.target[0]))**2 +
                              (float(agent.position[1]) - float(agent.target[1]))**2)
                obs[8] = min(dist / max(1, self.env.width + self.env.height), 1.0)
        except Exception as e:
            pass

        return obs.astype(np.float32)
