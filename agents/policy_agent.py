import os
import numpy as np
import torch
from game.agent import Agent
from game.training_data import state_to_features


class PolicyAgent(Agent):
    """This bot uses the policy to make decisions"""

    def __init__(self, policy=None, greedy=False, policy_path=None):
        self.policy = policy
        self.greedy = greedy
        self.policy_path = policy_path

        # set policy
        if self.policy is None:
            self.reset()

    def __repr__(self):
        return f"PolicyAgent(greedy={self.greedy})"

    def reset(self):
        self._load_from_path()

    def _load_from_path(self):
        """Load policy from path"""
        if self.policy_path is not None:
            assert os.path.exists(self.policy_path), 'File not found'
            self.policy = torch.load(self.policy_path)
            assert self.policy is not None, 'Failed to load policy'

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """Sample a random action according to the
        probability distribution of the policy"""
        assert self.policy is not None
        features = state_to_features(state_info)
        x = np.array([features])
        x = torch.tensor(x, dtype=torch.float)
        y = self.policy(x).detach().numpy()
        y *= state_info["legal_moves"]
        assert np.sum(y) > 0, "No legal moves"

        if self.greedy:
            action = int(np.argmax(y[0]))
        else:
            action = np.random.choice(len(y[0]), p=y[0] / np.sum(y[0]))  # todo normalization?

        return {"action": action, "eval": np.ones(2) * 0.5}
