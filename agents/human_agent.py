import numpy as np
from game.agent import Agent


class HumanAgent(Agent):

    def __repr__(self):
        return f"HumanAgent()"

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """Takes as input the current state info (like board, legal_moves, etc...).

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
        if len(move_indices) == 1:
            action = move_indices[0]
            print(f'legal moves: {move_indices}')
            print(f'only one move available: {action}')
            return {"action": action, "eval": np.ones(2) * 0.5}
        else:
            print(f'legal moves: {move_indices}')
            while True:
                try:
                    action = int(input('Enter your move: '))
                    assert action in move_indices, f'invalid move {action}'
                    break
                except AssertionError:
                    pass
            assert action in move_indices, f'invalid move {action}'
            return {"action": action, "eval": np.ones(2) * 0.5}
