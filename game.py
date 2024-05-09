from copy import deepcopy


class Game:
    def __init__(self, *args, **kwargs):
        ...

    def reset(self):
        ...

    def play(self, agents, verbose=False):
        ...

    def get_legal_moves(self):
        """
        returns a binary array-like object that represents
        the legal moves for the current player
        """
        ...

    def get_state_info(self):
        ...

    def is_game_over(self):
        ...

    def get_reward(self):
        ...

    def deepcopy(self):
        return deepcopy(self)
