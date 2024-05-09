import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from training import Training
from nn_agent import NNAgent
from royal_game_of_ur import RoyalGameOfUr
from training_data import create_dataset_from_game_files
from text_example_states import STATES


def show_self_play_data(path='database'):
    files = os.listdir(path)

    dct = dict()
    for i in range(len(files)):
        s = files[i]
        s = s.replace('game_', '')
        s = s.replace('.pkl', '')
        dct[int(s)] = files[i]

    print(dct)

    print(files)
    for file_name in files:
        with open(f"{path}\\{file_name}", 'rb') as f:
            game_info = pickle.load(f)

        print(f"\n{file_name}")
        if 'players' in game_info:
            print(f"{game_info['players']}")


def test_training_loop(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):

    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'models'),
                    n_rollouts=100,
                    rollout_depth=10)

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=root_dir + 'ur_games'
                        )

    # load agents
    training._load_agents()
    # print(training.agents)

    # self-play
    # training._play_self_play_games(n_games=1, verbose=True)

    # convert games to training data
    training._convert_games_to_training_data()

    # train policy and value function
    training._train_agent(n_epochs=5000)


def test_evaluation(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):

    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=100,
                    rollout_depth=None)
    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    training._load_agents()
    training._convert_games_to_training_data()

    game_dir = root_dir + 'ur_games'
    files = os.listdir(game_dir)
    file_id = np.random.choice(range(len(files)))
    file_name = game_dir + '\\' + files[file_id]
    X, y_policy, y_value = create_dataset_from_game_files(game_files=[file_name], halflife=0)

    y_pred = training.agent_instance.value_function(X)
    y_pred = y_pred.detach().numpy()

    plt.title(f'Evaluation of {file_name}')
    plt.plot(y_value)
    plt.plot(y_pred, alpha=0.5)

    plt.show()


def test_agent_decisions(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=100,
                    rollout_depth=5)

    for i, state in enumerate(STATES):
        print(f'\nExample {i})\n')
        game = RoyalGameOfUr().set_state(state)
        print(game)
        state_info = game.get_state_info()
        agent.get_action(state_info, verbose=True)


if __name__ == '__main__':
    # test_training(n_rollouts=40, rollout_depth=None)
    # show_self_play_data()
    # test_training_loop()
    # test_evaluation()
    test_agent_decisions()
