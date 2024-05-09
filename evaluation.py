import numpy as np
from royal_game_of_ur import RoyalGameOfUr
from agent import Agent
from nn_agent import NNAgent
import os

def play_game(agent_1, agent_2, verbose):
    game = RoyalGameOfUr()
    result = game.play(agents=(agent_1, agent_2),
                       verbose=verbose)
    return result["reward"]


def evaluation_match(agent_1, agent_2, n_games, show_game=False, verbose=True):
    out = 0
    for i in range(n_games):
        out += play_game(agent_1, agent_2, verbose=show_game)
        if verbose:
            print(f'game: {i+1:3} / {n_games:3}   score: {np.round(out / n_games, 3)}')
    return out / n_games


def main(n_games=10, show_game=False, root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
    agent_1 = NNAgent(game_instance=RoyalGameOfUr(),
                      models_dir=os.path.join(root_dir, 'ur_models'),
                      n_rollouts=50,
                      rollout_depth=2)
    agent_2 = Agent()
    evaluation_match(agent_1, agent_2, n_games=n_games, show_game=show_game)
    print()

    # swap colors
    agent_1, agent_2 = agent_2, agent_1
    evaluation_match(agent_1, agent_2, n_games=n_games, show_game=show_game)


if __name__ == '__main__':
    main()
