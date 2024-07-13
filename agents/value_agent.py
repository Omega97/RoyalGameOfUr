import os
import numpy as np
import torch
from game.agent import Agent
from game.training_data import state_to_features


class ValueAgent(Agent):
    """
    This bot uses the value function to make decisions
    """
    def __init__(self, game_instance, value_function=None, value_path=None, verbose=False):
        self.game_instance = game_instance
        self.value_function = value_function
        self.value_path = value_path
        self.verbose = verbose

        # set value function
        if self.value_function is None:
            self.reset()

    def __repr__(self):
        return f"ValueAgent()"

    def reset(self):
        if self.verbose:
            print('Loading value function')
        self._load_from_path()

    def _load_from_path(self):
        """Load value function from path"""
        assert self.value_path is not None
        assert os.path.exists(self.value_path), 'File not found'
        self.value_function = torch.load(self.value_path)
        assert self.value_function is not None, 'Failed to load value function'
        if self.verbose:
            print('Value function loaded')

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """
        Evaluate all the possible moves and return the one
        with the highest expected value.
        """
        assert self.value_function is not None
        player_id = state_info["current_player"]
        legal_moves = state_info["legal_moves"]
        legal_move_indices = np.arange(len(legal_moves))[legal_moves > 0]
        values = np.zeros(len(legal_moves))

        for i in range(len(legal_move_indices)):
            state = self.game_instance.deepcopy()
            state = state.set_state(state_info)
            state.move(legal_move_indices[i])
            features = state_to_features(state.get_state_info())
            x = np.array([features])
            x = torch.tensor(x, dtype=torch.float)
            y = self.value_function(x).detach().numpy()
            y = y[0]
            values[i] = y[player_id]

        # get best move
        best_move = legal_move_indices[np.argmax(values)]
        best_eval = max(values)
        eval_ = [best_eval, 1-best_eval]
        if player_id == 1:
            eval_ = list(reversed(eval_))
        eval_ = np.array(eval_)

        if self.verbose:
            print(f'action {best_move}')
            print(f'  eval {best_eval:.3f}')

        return {"action": best_move, "eval": eval_}
