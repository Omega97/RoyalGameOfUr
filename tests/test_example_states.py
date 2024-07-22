import numpy as np
import torch
from example_states_test import STATES
from game.training_data import state_to_features
from game.royal_game_of_ur import RoyalGameOfUr
from agents.nn_value_agent import NNValueAgent


def test_eval_all_states():

    x = []

    for i in range(len(STATES)):
        state = RoyalGameOfUr().set_state(STATES[i])
        x.append(state_to_features(state.get_state_info()))

    x = torch.tensor(np.array(x), dtype=torch.float32)

    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path='../ur_models',
                         )

    y = agent.value_function(x).detach().numpy()
    print(np.round(y, 3))


def test_eval_initial_states():

    x = []
    info_0 = STATES[-1]

    for i in range(5):
        info = info_0
        info['n_steps'] = i
        state = RoyalGameOfUr().set_state(info)
        x.append(state_to_features(state.get_state_info()))

    x = torch.tensor(np.array(x), dtype=torch.float32)

    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path='../ur_models',
                         )

    y = agent.value_function(x).detach().numpy()[:, 0]
    w_0 = np.array([1, 4, 6, 4, 1]) / 16
    w_1 = np.array([0, 4, 6, 4, 1]) / 15

    value_0 = y[0]
    value_1 = y.dot(w_1)
    value_2 = (16-15*value_1)/17
    value_avg = y.dot(w_0)

    print()
    for i in range(len(y)):
        print(f'{i})  {y[i]:.2%}')

    print()
    print(f'avg_val = {value_avg:.2%}')

    print()
    print(f'value_0 = {value_0:6.2%}')
    print(f'          {value_2:6.2%}')
    print(f'          {value_0 - value_2:+6.2%}   <-  (should be 0%)')


def test_get_action():
    state = RoyalGameOfUr().set_state(STATES[0])
    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path='../ur_models',
                         )

    print(state)
    print(agent.get_action(state.get_state_info()))


if __name__ == '__main__':
    # test_eval_all_states()
    test_eval_initial_states()
    # test_get_action()
