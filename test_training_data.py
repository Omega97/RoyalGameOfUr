import numpy as np
import matplotlib.pyplot as plt
import os
from training_data import backpropagation, create_dataset_from_game_files


def test_backpropagation(n=100):
    x_ = np.linspace(0, 1, n)

    # y_ = np.cos(x_ * np.pi * 4) + np.random.normal(0, 0.1, n)
    y_ = np.zeros(n)
    y_[-1] = 1

    out = backpropagation(y_, halflife=10)
    plt.title('Backpropagation')
    plt.plot(x_, y_, label='Original')
    plt.plot(x_, out, label='Backpropagation')
    plt.legend()
    plt.show()


def test_create_dataset(games_dir='database'):
    file_list = os.listdir(games_dir)
    file_list = [os.path.join(games_dir, name) for name in file_list if name.endswith('.pkl')]

    file_list = file_list[:3]

    print(f'Games found: {len(file_list)}')

    x, y_policy, y_value = create_dataset_from_game_files(file_list, halflife=10)
    plt.plot(y_value)

    x, y_policy, y_value = create_dataset_from_game_files(file_list, halflife=0)
    plt.plot(y_value, alpha=0.4)

    plt.show()


if __name__ == '__main__':
    # test_backpropagation()
    test_create_dataset()
