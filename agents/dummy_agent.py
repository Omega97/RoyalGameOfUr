import numpy as np
from game.agent import Agent


class DummyAgent(Agent):
    """Handcrafted bot (elo = 370)"""
    def __init__(self, policy=None, greedy=False, policy_path=None):
        self.policy = policy
        self.greedy = greedy
        self.policy_path = policy_path

        # set policy
        if self.policy is None:
            self.reset()

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """move the last piece in the list of legal moves"""
        moves = state_info["legal_move_indices"]
        return {"action": moves[-1], "eval": np.ones(2) * 0.5}


class DummyAgent2(DummyAgent):
    """Handcrafted bot (elo = 480 = DummyAgent + 110)"""
    def get_action(self, state_info: dict, verbose=False) -> dict:
        """move the last piece in the list of legal moves,
        but tries to take opponent's pieces"""
        moves = state_info["legal_move_indices"]
        roll = state_info["n_steps"]
        opp_board = state_info["board"][1 - state_info["current_player"]]
        move = moves[-1]
        if move <= 3:
            move = moves[0]
        if len(moves) > 1:
            x = moves[-2] + roll
            if opp_board[x]:
                move = moves[-2]
        return {"action": move, "eval": np.ones(2) * 0.5}
