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
    print(x.dtype)

    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path='../ur_models',
                         depth=2,
                         )

    y = agent.value_function(x).detach().numpy()
    print(np.round(y, 3))


def test_get_action():
    state = RoyalGameOfUr().set_state(STATES[0])
    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path='../ur_models',
                         depth=2,
                         )

    print(state)
    print(agent.get_action(state.get_state_info()))


if __name__ == '__main__':
    # test_eval_all_states()
    test_get_action()
