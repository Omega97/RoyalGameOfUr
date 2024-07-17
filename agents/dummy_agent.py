import numpy as np
from game.agent import Agent


class DummyAgent(Agent):
    """Handcrafted bot (elo=370)"""

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


class GreedyAgent(DummyAgent):
    """Handcrafted bot (elo=??)"""

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """move the last piece in the list of legal moves"""
        legal_moves = state_info["legal_moves"]
        scores = np.array(legal_moves, dtype=float)

        scores *= np.arange(1, len(scores) + 1) / len(scores)

        board = np.array(state_info["board"])
        roll = state_info["n_steps"]
        player = state_info["current_player"]

        my_board = board[player]
        opp_board = board[1-player]
        for i in range(5, 13):
            if legal_moves[i]:
                if my_board[i-roll]:
                    if opp_board[i]:
                        scores[i] += 1

        move = np.argmax(scores)

        # print(player)
        # print(board)
        # print(np.round(scores, 2))
        # print(roll)
        # print(move)
        # input()

        return {"action": move, "eval": np.ones(2) * 0.5}

