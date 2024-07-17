import numpy as np
from time import time
from game.royal_game_of_ur import RoyalGameOfUr
from game.evaluation import evaluation_match
from agents.random_agent import RandomAgent
from agents.nn_value_agent import NNValueAgent


def main(n_games=50):
    np.random.seed(0)

    agent_1 = NNValueAgent(game_instance=RoyalGameOfUr(), models_dir_path='../ur_models')
    agent_2 = RandomAgent()

    t = time()
    evaluation_match(agent_1, agent_2, n_games=n_games, player=0)
    evaluation_match(agent_2, agent_1, n_games=n_games, player=1)
    t = time() - t

    print(f'\n time = {t:.2f} s')


if __name__ == '__main__':
    main()
