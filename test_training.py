import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from training import Training
from nn_agent import NNAgent
from royal_game_of_ur import RoyalGameOfUr
from training_data import create_dataset_from_game_files


def show_self_play_data(path='database'):
    """Show the data stored in the database."""
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
    """Train the agent"""
    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'models'),
                    n_rollouts=100,
                    rollout_depth=10)

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games')
                        )

    # load agents
    training._load_agents()
    # print(training.agents)

    # self-play
    # training._play_self_play_games(n_games=1, verbose=True)

    # convert games to training data
    training.convert_games_to_training_data()

    # train policy and value function
    training._train_agent(n_epochs_policy=500,
                          n_epochs_value=500,
                          lr=0.01)


def test_evaluation(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
    """Evaluate the agent"""
    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=100,
                    rollout_depth=None)
    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    training._load_agents()
    training.convert_games_to_training_data()

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


def test_policy_training(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
    """Train the policy network"""
    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'))

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    # agent.reset_policy(input_size=82, hidden_units=16, output_size=16)
    input_size = 82
    hidden_units = 100
    output_size = 16
    model = (
        nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.Sigmoid(),
            nn.Linear(hidden_units, output_size),
            nn.Softmax(dim=1)
        ))

    training.convert_games_to_training_data()

    print(training.X.shape)
    print(training.y_policy.shape)

    # accuracy before
    y_true = training.y_policy.detach().numpy()
    y_true_values = np.argmax(y_true, axis=1)

    y_pred = agent.policy(training.X)
    y_pred_values = np.argmax(y_pred.detach().numpy(), axis=1)
    accuracy = np.mean(y_pred_values == y_true_values)
    dummy_ans = np.argmax(y_true.sum(axis=0))
    dummy_acc = np.mean(dummy_ans == y_true_values)
    print(f'\ndummy acc: {dummy_acc:.1%}')
    print(f' accuracy: {accuracy:.2%}')

    agent.train_policy(training.X, training.y_policy)

    # accuracy after
    y_pred = agent.policy(training.X)
    y_pred_values = np.argmax(y_pred.detach().numpy(), axis=1)
    accuracy = np.mean(y_pred_values == y_true_values)

    print()
    for i in range(30):
        j = -1-i
        s = "".join(['#' if i == 1 else '_' for i in training.X[j]])
        print(f'{y_true_values[j]:2} {y_pred_values[j]:2}  {s}')
    print()

    print(f'\ndummy acc: {dummy_acc:.1%}')
    print(f' accuracy: {accuracy:.2%}')

    # agent.save_models()


def test_policy(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):
    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'))
    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    training.convert_games_to_training_data()
    y_pred = agent.policy(training.X).detach().numpy()
    y_pred_max = np.max(y_pred, axis=1)

    plt.hist(y_pred_max, bins=20)
    plt.show()


if __name__ == '__main__':
    # test_training(n_rollouts=40, rollout_depth=None)
    # show_self_play_data()
    # test_training_loop()
    # test_evaluation()
    # test_policy_training()
    test_policy()
