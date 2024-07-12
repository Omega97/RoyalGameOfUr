import numpy as np
from game.royal_game_of_ur import RoyalGameOfUr
from agents.nn_agent import PolicyAgent, NNAgent, create_policy, create_value_function


def test_policy_agent():
    game = RoyalGameOfUr()
    policy = create_policy()
    agents = [PolicyAgent(policy), PolicyAgent(policy)]
    game.play(agents, verbose=True)


def test_nn_agent_state():
    state_info = {'current_player': 0,
                  'n_steps': 2,
                  'round': 244,
                  'board': np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]]),
                  }
    game = RoyalGameOfUr()
    game.set_state(state_info)
    print(game)

    policy = create_policy()
    value_function = create_value_function()
    agent = NNAgent(policy, value_function, game, n_rollouts=50, rollout_depth=20)
    output = agent(game.get_state_info(), verbose=True)
    game.move(output["action"])
    print(game)


def test_nn_agent(n_rollouts=100, rollout_depth=None):
    state_info = {'current_player': 0,
                  'n_steps': 1,
                  'round': 244,
                  'board': np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 4]]),
                  }
    game = RoyalGameOfUr()
    game.set_state(state_info)

    policy = create_policy()
    value_function = create_value_function()
    agents = [
        NNAgent(policy, value_function, game, n_rollouts=n_rollouts, rollout_depth=rollout_depth),
        NNAgent(policy, value_function, game, n_rollouts=n_rollouts, rollout_depth=rollout_depth),
    ]
    game.play(agents, verbose=True, do_reset=True)


if __name__ == '__main__':
    # test_policy_agent()
    # test_nn_agent_state()
    test_nn_agent(n_rollouts=30, rollout_depth=None)
