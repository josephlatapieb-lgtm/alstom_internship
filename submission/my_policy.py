import numpy as np

class MyPolicy:
    def __init__(self):
        pass

    def act(self, obs, env=None):
        return np.random.randint(0, 5)
