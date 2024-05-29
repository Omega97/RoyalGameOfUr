import os
import numpy as np
from agent import Agent
from royal_game_of_ur import RoyalGameOfUr
from nn_agent import NNAgent
import matplotlib.pyplot as plt


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
            return {"action": action}
        else:
            print(f'legal moves: {move_indices}')
            while True:
                try:
                    action = int(input('Enter your move: '))
                    assert action in move_indices
                    break
                except Exception:
                    pass
            assert action in move_indices, f'invalid move {action}'
            return {"action": action}


def play_game(agent_1, agent_2, verbose):
    game = RoyalGameOfUr()
    game_recap = game.play(agents=(agent_1, agent_2),
                           verbose=verbose)
    return game_recap


def plot_game(game_recap):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(len(game_recap["player_eval"])):
        v1, v2 = game_recap["player_eval"][i]
        if v1 is not None:
            x1.append(i)
            y1.append(v1)
        if v2 is not None:
            x2.append(i)
            y2.append(v2)

    plt.title('Game evaluation')
    plt.plot(x1, y1, label='player 1', alpha=0.3)
    plt.plot(x2, y2, label='player 2', alpha=0.3)
    plt.scatter(x1, y1, s=3)
    plt.scatter(x2, y2, s=3)
    plt.xlabel('turn')
    plt.ylabel('evaluation')
    plt.legend()
    plt.show()


def main(n_rollouts=200, rollout_depth=5, root_dir=os.getcwd()):
    """Play a game between two agents and print the result."""
    agents = list()
    # agents.append(HumanAgent())

    agents.append(NNAgent(game_instance=RoyalGameOfUr(),
                          dir_path=os.path.join(root_dir, 'ur_models'),
                          n_rollouts=n_rollouts,
                          rollout_depth=rollout_depth,
                          verbose=True))

    agents.append(NNAgent(game_instance=RoyalGameOfUr(),
                          dir_path=os.path.join(root_dir, 'ur_models'),
                          n_rollouts=n_rollouts,
                          rollout_depth=rollout_depth,
                          verbose=True))

    # policy_path = os.path.join(root_dir, 'ur_models\\policy.pkl')
    # agents.append(PolicyAgent(policy_path=policy_path, greedy=True))

    # agents.append(Agent())
    # agents.append(Agent())

    # agents = list(reversed(agents))

    assert len(agents) == 2

    recap = play_game(*agents, verbose=True)
    for k in ["players", "rolls", "player_moves", "reward"]:
        print(f"{k:>12}:  {recap[k]}")

    plot_game(recap)


if __name__ == '__main__':
    main()
