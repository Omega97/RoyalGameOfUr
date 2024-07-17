import numpy as np
import matplotlib.pyplot as plt
from game.royal_game_of_ur import RoyalGameOfUr
from agents.random_agent import RandomAgent
from agents.value_agent import ValueAgent
from agents.nn_value_agent import NNValueAgent
from agents.human_agent import HumanAgent
# from agents.dummy_agent import DummyAgent


def plot_game(game_recap):
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


def play_game(agent_1, agent_2, verbose=True, do_plot=True):
    game = RoyalGameOfUr()
    game_recap = game.play(agents=(agent_1, agent_2),
                           verbose=verbose)
    for k in ["players", "rolls", "player_moves", "player_on_duty", "reward"]:
        s = " ".join([str(e) for e in game_recap[k]])
        print(f"{k:>15}:  {s}")

    if do_plot:
        plot_game(game_recap)


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
    agents.append(NNValueAgent(game_instance=RoyalGameOfUr(), models_dir_path=dir_path))
    agents.append(HumanAgent())
    assert len(agents) == 2
    play_game(*agents, do_plot=True)


def demo_game_nn_value_agent(dir_path='..//ur_models'):
    """Play a game between two NN-based agents and print the result"""
    # np.random.seed(13)  # 13 0 1 14 3 2
    agents = list()
    agents.append(NNValueAgent(game_instance=RoyalGameOfUr(), models_dir_path=dir_path))
    agents.append(NNValueAgent(game_instance=RoyalGameOfUr(), models_dir_path=dir_path))
    assert len(agents) == 2
    play_game(*agents, do_plot=True)


if __name__ == '__main__':
    # demo_game_random()
    # demo_game_value_agent()
    # demo_game_nn_value_vs_human()
    demo_game_nn_value_agent()
