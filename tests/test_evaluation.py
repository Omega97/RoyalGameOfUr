from time import time
from game.royal_game_of_ur import RoyalGameOfUr
from game.evaluation import evaluation_match
from agents.nn_value_agent import DeepNNValueAgent
from agents.dummy_agent import DummyAgent2
from agents.random_agent import RandomAgent


def match(agent1, agent2, n_games=200, verbose=True):

    t = time()
    count_1 = evaluation_match(agent1, agent2, n_games=n_games, verbose=verbose, player=0)
    count_2 = evaluation_match(agent2, agent1, n_games=n_games, verbose=verbose, player=1)
    count = count_1 - count_2
    t = time() - t

    if verbose:
        print(f'\n time = {t:.2f} s\n')
    return count



def match_deep_nn(agent_names, n_games=200, verbose=True):

    assert len(agent_names) == 2
    agents = list()

    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                               models_dir_path='../ur_models',
                               value_function_name=agent_names[0]))

    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                               models_dir_path='../ur_models',
                               value_function_name=agent_names[1]))

    return match(*agents, n_games=n_games, verbose=verbose)


if __name__ == '__main__':
    match(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                           models_dir_path='../ur_models',
                           value_function_name='value_function.pkl'),
          DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                           models_dir_path='../ur_models',
                           value_function_name='softplus_100_100.pkl'),
          n_games=100)
    # match_deep_nn(('softplus_160_160_80.pkl', 'softplus_100_100.pkl'))
