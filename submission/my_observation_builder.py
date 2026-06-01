import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder

class MyObservationBuilder(ObservationBuilder):
    def reset(self):
        return True

    def get(self, handle):
        return np.zeros(36, dtype=np.float32)
