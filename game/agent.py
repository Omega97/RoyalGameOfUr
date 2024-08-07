import numpy as np


class Agent:

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """
        Takes as input the current state info (like board, legal_moves, etc...).

        Returns a dictionary containing the action to take (as a number between 0 and n_actions)
        and other info:
        - action: the action to take (*mandatory)
        - eval: the evaluation of the state (*optional)

        Input example:
        state_info = {"current_player": 0,
                      "n_steps": 2,
                      "round": 9,
                      "board": np.array([7, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0,
                                         7, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0]),
                      "legal_moves": np.array([1., 0., 1., 0., 0., 0., 0., 0.,
                                               0., 0., 0., 0., 0., 0., 0., 0.])
                                               }

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
        return {"action": action, "eval": [0.5, 0.5]}

    def reset(self):
        """Resets the agent to its initial state before the game."""
        pass

    def __call__(self, state_info: dict, **kw) -> dict:
        """
        Returns the agent's action on a given state
        as a dictionary of "action" and "eval"
        """
        assert type(state_info) is dict
        out = self.get_action(state_info, **kw)
        assert type(out) is dict
        for key in ('action', 'eval'):
            assert key in out, f'{self} {out}'
        return out

    def __repr__(self):
        """Name of the agent"""
        return f"{type(self).__name__}()"
