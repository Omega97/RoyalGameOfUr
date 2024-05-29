import numpy as np
import os
from time import time
from royal_game_of_ur import RoyalGameOfUr
# from nn_agent import NNAgent, PolicyAgent
from agent import Agent
from example_states_test import STATES


def elo(winrate, k=400):
    """Calculate the Elo rating from a winrate"""
    return k * np.log10(winrate / (1 - winrate))


def play_game(agent_1, agent_2, verbose):
    game = RoyalGameOfUr()
    result = game.play(agents=(agent_1, agent_2),
                       verbose=verbose)
    return result["reward"]


def evaluation_match(agent_1, agent_2, n_games,
                     show_game=False, prior=0.5,
                     player=0, verbose=True):
    out = np.zeros(2, dtype=float)
    score = None
    times = [time()]
    for i in range(n_games):
        out += play_game(agent_1, agent_2, verbose=show_game)
        score = (out + prior) / (i + 1 + 2 * prior)
        times.append(time())
        if verbose:
            e = elo(score)
            j_start = (len(times) - 1) // 2
            j_stop = len(times) - 1
            n_left = n_games - j_stop
            speed = (j_stop - j_start) / (times[j_stop] - times[j_start])
            eta = n_left / speed
            print(f'\rgame: {i + 1:3} / {n_games:3}   '
                  f'score: {score[player]:.3f}   '
                  f'elo: {e[player]:.0f}   '
                  f'eta: {eta // 60:.0f} min {eta % 60:.0f} s', end='  ')
    print()
    return score


# def test_agent_decisions(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
#     """Test the agent's decisions on some example states"""
#     agent = NNAgent(game_instance=RoyalGameOfUr(),
#                     models_dir=os.path.join(root_dir, 'ur_models'),
#                     n_rollouts=200,
#                     rollout_depth=5)
#
#     states = STATES
#
#     for i, state in enumerate(states):
#         print(f'\nExample {i})\n')
#         game = RoyalGameOfUr().set_state(state)
#         print(game)
#         state_info = game.get_state_info()
#         agent.get_action(state_info, verbose=True)


# def test_evaluation_game(n_rollouts=300, rollout_depth=5,
#                          root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
#     """Play a game between two agents and print the result."""
#     agent_1 = NNAgent(game_instance=RoyalGameOfUr(),
#                       models_dir=os.path.join(root_dir, 'ur_models'),
#                       n_rollouts=n_rollouts,
#                       rollout_depth=rollout_depth)
#
#     agent_2 = NNAgent(game_instance=RoyalGameOfUr(),
#                       models_dir=os.path.join(root_dir, 'ur_models'),
#                       n_rollouts=n_rollouts,
#                       rollout_depth=rollout_depth)
#
#     play_game(agent_1, agent_2, verbose=True)


def main(n_games=100, n_rollouts=100, rollout_depth=5, show_game=False,
         root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):

    agents = list()

    # agents.append(NNAgent(game_instance=RoyalGameOfUr(),
    #                       models_dir=os.path.join(root_dir, 'ur_models'),
    #                       n_rollouts=n_rollouts,
    #                       rollout_depth=rollout_depth))

    # agents.append(NNAgent(game_instance=RoyalGameOfUr(),
    #                       models_dir=os.path.join(root_dir, 'ur_models'),
    #                       n_rollouts=n_rollouts,
    #                       rollout_depth=rollout_depth))

    # value_path = os.path.join(root_dir, 'ur_models\\value_function.pkl')
    # agents.append(ValueAgent(game_instance=RoyalGameOfUr(), value_path=value_path))

    # policy_path = os.path.join(root_dir, 'ur_models\\policy.pkl')
    # agents.append(PolicyAgent(policy_path=policy_path, greedy=True))
    # agents.append(Agent())
    #
    # assert len(agents) == 2
    #
    # print()
    # print('Agent 1:', agents[0])
    # print('Agent 2:', agents[1])
    # print()
    #
    # # evaluation match
    # evaluation_match(*agents, n_games=n_games, show_game=show_game, player=0)
    # print()
    #
    # # swap colors
    # agents = list(reversed(agents))
    # evaluation_match(*agents, n_games=n_games, show_game=show_game, player=1)
    # print()


if __name__ == '__main__':
    # test_agent_decisions()
    main(n_games=500, rollout_depth=5)
