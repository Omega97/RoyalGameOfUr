import numpy as np
import matplotlib.pyplot as plt
from game.royal_game_of_ur import RoyalGameOfUr
from agents.random_agent import RandomAgent
from agents.value_agent import ValueAgent
from agents.nn_value_agent import NNValueAgent, DeepNNValueAgent
from agents.human_agent import HumanAgent
# from agents.dummy_agent import DummyAgent


def plot_game(game_recap, title='Game evaluation', show=True):
    n_rounds = len(game_recap["player_eval"])
    v = np.array(game_recap["player_on_duty"], dtype=int)
    x1 = list(np.arange(n_rounds)[v == 0])
    x2 = list(np.arange(n_rounds)[v == 1])
    y1 = [game_recap["player_eval"][t][0] for t in x1]
    y2 = [game_recap["player_eval"][t][1] for t in x2]
    x1.append(n_rounds)
    x2.append(n_rounds)
    y1.append(game_recap["reward"][0])
    y2.append(game_recap["reward"][1])

    plt.title(title)
    plt.plot(x1, y1, label=game_recap["players"][0], alpha=0.5)
    plt.plot(x2, y2, label=game_recap["players"][1], alpha=0.5)
    plt.scatter(x1, y1, s=6)
    plt.scatter(x2, y2, s=6)
    # plt.xlabel('turn')
    # plt.ylabel('evaluation')
    plt.legend()
    plt.ylim(0, 1)
    if show:
        plt.show()


def play_game(agent_1, agent_2, verbose=True, do_plot=True, title='Game evaluation', show=True):
    game = RoyalGameOfUr()
    game_recap = game.play(agents=(agent_1, agent_2),
                           verbose=verbose)
    for k in ["players", "rolls", "player_moves", "player_on_duty", "n_legal_moves", "reward"]:
        s = " ".join([str(e) for e in game_recap[k]])
        print(f"{k:>15}:  {s}")

    if do_plot:
        plot_game(game_recap, title=title, show=show)


def demo_game_random():
    """Play a game between two random agents and print the result"""
    agents = list()
    agents.append(RandomAgent())
    agents.append(RandomAgent())
    assert len(agents) == 2
    play_game(*agents, do_plot=False)


def demo_game_value_agent(value_path='..data/ur_models/model.pkl'):
    """Play a game between two NN-based agents and print the result"""
    agents = list()
    agents.append(ValueAgent(game_instance=RoyalGameOfUr(), value_path=value_path))
    agents.append(ValueAgent(game_instance=RoyalGameOfUr(), value_path=value_path))
    assert len(agents) == 2
    play_game(*agents, do_plot=False)


def demo_game_nn_value_vs_human(dir_path='..//ur_models'):
    """Play a game between two NN-based agents and print the result"""
    agents = list()

    agents.append(HumanAgent())
    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=3,
                                   models_dir_path=dir_path,
                                   value_function_name='softplus_100_100.pkl'))

    assert len(agents) == 2
    play_game(*agents, do_plot=True)


def demo_game_nn_value_agent(dir_path='..//ur_models', seed=0, swap=False,
                             verbose=True, title='Game evaluation', show=True):
    """Play a game between two NN-based agents and print the result"""
    np.random.seed(seed)  # 0) 13  1) 0  2) 1  3) 3  4) 2
    agents = list()
    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                                   models_dir_path=dir_path, value_function_name='value_function.pkl'))
    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                                   models_dir_path=dir_path, value_function_name='value_function.pkl'))
    # agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
    #                                models_dir_path=dir_path, value_function_name='softplus_100_100.pkl'))
    # agents.append(NNValueAgent(game_instance=RoyalGameOfUr(), models_dir_path=dir_path))
    assert len(agents) == 2
    if swap:
        agents = list(reversed(agents))
    play_game(*agents, verbose=verbose, do_plot=True,title=title, show=show)


def game_plots(nrows=2, ncols=3, swap=False):
    seed = 0
    fig, ax = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            plt.sca(ax[i, j])
            print()
            demo_game_nn_value_agent(seed=seed, verbose=False, swap=swap,
                                     title=f'Seed = {seed}', show=False)
            seed += 1
    plt.show()


if __name__ == '__main__':
    # demo_game_random()
    # demo_game_value_agent()
    # demo_game_nn_value_vs_human()
    demo_game_nn_value_agent()
    # game_plots()
    # game_plots(swap=True)
