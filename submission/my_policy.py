import numpy as np
import os

class SimplePolicy:
    def __init__(self):
        self.weights = np.random.randn(36, 5) * 0.01
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.npy")
        if os.path.exists(checkpoint_path):
            try:
                self.weights = np.load(checkpoint_path)
            except:
                pass

    def act(self, obs):
        obs = np.array(obs, dtype=np.float32)
        logits = obs @ self.weights
        return np.argmax(logits).item()

class MyPolicy:
    def __init__(self):
        self.policy = SimplePolicy()

    def act(self, obs, env=None):
        return self.policy.act(obs)
