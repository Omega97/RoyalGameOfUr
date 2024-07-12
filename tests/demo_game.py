import os
import numpy as np
from game.agent import Agent
from game.royal_game_of_ur import RoyalGameOfUr
from agents.nn_agent import NNAgent, PolicyAgent, ValueAgent
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
            return {"action": action, "eval": np.ones(2) * 0.5}
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
            return {"action": action, "eval": np.ones(2) * 0.5}


def play_game(agent_1, agent_2, verbose):
    game = RoyalGameOfUr()
    game_recap = game.play(agents=(agent_1, agent_2),
                           verbose=verbose)
    return game_recap


def plot_game(game_recap):
    n_rounds = len(game_recap["player_eval"])
    v = np.array(game_recap["player_on_duty"], dtype=int)
    x1 = list(np.arange(n_rounds)[v==0])
    x2 = list(np.arange(n_rounds)[v==1])
    y1 = [game_recap["player_eval"][t][0] for t in x1]
    y2 = [game_recap["player_eval"][t][1] for t in x2]
    x1.append(n_rounds)
    x2.append(n_rounds)
    y1.append(game_recap["reward"][0])
    y2.append(game_recap["reward"][1])

    plt.title('Game evaluation')
    plt.plot(x1, y1, label=game_recap["players"][0], alpha=0.5)
    plt.plot(x2, y2, label=game_recap["players"][1], alpha=0.5)
    plt.scatter(x1, y1, s=6)
    plt.scatter(x2, y2, s=6)
    plt.xlabel('turn')
    plt.ylabel('evaluation')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def main(n_rollouts=100, rollout_depth=3, root_dir=os.getcwd()):
    """Play a game between two agents and print the result."""
    agents = list()

    # human
    # agents.append(HumanAgent())

    agents.append(NNAgent(game_instance=RoyalGameOfUr(),
                          dir_path=os.path.join(root_dir, 'ur_models'),
                          n_rollouts=n_rollouts,
                          rollout_depth=rollout_depth,
                          verbose=True))

    # agents.append(NNAgent(game_instance=RoyalGameOfUr(),
    #                       dir_path=os.path.join(root_dir, 'ur_models'),
    #                       n_rollouts=n_rollouts * 3,
    #                       rollout_depth=rollout_depth,
    #                       verbose=True))

    policy_path = os.path.join(root_dir, 'ur_models\\policy.pkl')
    agents.append(PolicyAgent(policy_path=policy_path, greedy=True))

    # value_path = os.path.join(root_dir, 'ur_models\\value_function.pkl')
    # agents.append(ValueAgent(game_instance=RoyalGameOfUr(),
    #                          value_path=value_path,
    #                          verbose=True))

    # agents.append(Agent())
    # agents.append(Agent())

    # swap players
    # agents = list(reversed(agents))

    assert len(agents) == 2

    recap = play_game(*agents, verbose=True)
    for k in ["players", "rolls", "player_moves", "player_on_duty", "reward"]:
        print(f"{k:>15}:  {recap[k]}")

    plot_game(recap)


if __name__ == '__main__':
    main()
