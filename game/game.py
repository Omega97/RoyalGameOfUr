import numpy as np
from copy import deepcopy


class Game:

    def __init__(self, *args, **kwargs):
        ...

    def reset(self):
        """ Reset the game to avoid having to create a new instance every time """
        ...

    def play(self, agents, verbose=False):
        """ Play a game between the agents """
        ...

    def get_legal_moves(self):
        """
        returns a binary array-like object that represents
        the legal moves for the current player
        """
        ...

    def get_state_info(self) -> dict:
        """
        Return a dictionary containing info about
        states current_player, round, board info, legal moves
        """
        ...

    def is_game_over(self) -> bool:
        """ Return True if game is over, else False """
        ...

    def get_reward(self) -> np.array:
        """ Returns a numpy array of rewards for each player when game is over """
        ...

    def deepcopy(self):
        """ Return a deep copy of the game """
        return deepcopy(self)
