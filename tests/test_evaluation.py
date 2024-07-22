from time import time
from game.royal_game_of_ur import RoyalGameOfUr
from game.evaluation import evaluation_match
from agents.nn_value_agent import NNValueAgent
from agents.dummy_agent import DummyAgent2


def main(n_games=100):
    agents = list()

    agents.append(NNValueAgent(game_instance=RoyalGameOfUr(),
                               models_dir_path='../ur_models',
                               value_function_name='value_function.pkl'))
    # agents.append(NNValueAgent(game_instance=RoyalGameOfUr(),
    #                            models_dir_path='../ur_models',
    #                            value_function_name='sigmoid_120_120.pkl'))
    agents.append(NNValueAgent(game_instance=RoyalGameOfUr(),
                               models_dir_path='../ur_models',
                               value_function_name='softplus_100_100.pkl'))
    # agents.append(DummyAgent2())

    assert len(agents) == 2

    t = time()
    evaluation_match(agents[0], agents[1], n_games=n_games, player=0)
    evaluation_match(agents[1], agents[0], n_games=n_games, player=1)
    t = time() - t

    print(f'\n time = {t:.2f} s')


if __name__ == '__main__':
    main()
