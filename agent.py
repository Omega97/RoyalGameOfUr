import numpy as np
from copy import deepcopy


class Agent:

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """
        Takes as input the current state info (like board, legal_moves, etc...).

        Returns a dictionary containing the action to take (as a number between 0 and n_actions)
        and other info:
        - action: the action to take (*mandatory)
        - eval: the evaluation of the state (*optional)

        Output example:
        {"action": 15, "eval": [0.5, 0.5]}
        """
        legal_moves = state_info["legal_moves"]
        assert legal_moves.sum() > 0, f'no legal moves available for {self}'

        move_indices = np.arange(len(legal_moves))[legal_moves > 0]
        action = np.random.choice(move_indices)
        if verbose:
            print('Agent()')
            legal_indices = np.arange(len(legal_moves))[legal_moves > 0]
            print(f'legal moves: {legal_indices}')
            print(f'action: {action}')
        return {"action": action}

    def __call__(self, state_info, **kw):
        out = self.get_action(state_info, **kw)
        assert type(out) is dict
        assert 'action' in out
        # assert 'eval' in out
        return out

    def __repr__(self):
        return f"Agent()"
