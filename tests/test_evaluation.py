import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from time import time
from game.royal_game_of_ur import RoyalGameOfUr
from game.evaluation import evaluation_match
from agents.nn_value_agent import DeepNNValueAgent


def match(agent_names, n_games=200, verbose=True):

    assert len(agent_names) == 2
    agents = list()

    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                               models_dir_path='../ur_models',
                               value_function_name=agent_names[0]))

    agents.append(DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=1,
                               models_dir_path='../ur_models',
                               value_function_name=agent_names[1]))
    t = time()
    count_1 = evaluation_match(agents[0], agents[1], n_games=n_games, verbose=verbose, player=0)
    count_2 = evaluation_match(agents[0], agents[1], n_games=n_games, verbose=verbose, player=1)
    count = count_1 + count_2
    t = time() - t

    if verbose:
        print(f'\n time = {t:.2f} s\n')
    return count - n_games


def tournament(models_dir_path='../ur_models',
               data_path='tournament.pkl',
               n_games=1, do_plot=False):

    # declare model names
    names = os.listdir(models_dir_path)

    # load data
    try:
        with open(data_path, 'rb') as file:
            mat = pickle.load(file)
    except FileNotFoundError:
        mat = np.zeros((len(names), len(names)))

    n = int(mat[0, 0])
    mat[0, 0] = 0

    while True:
        n += 1

        if n_games > 0:
            # run the tournament
            for i in range(len(names)):
                for j in range(i):
                    print(f'{names[i]} vs {names[j]}')
                    c = match(agent_names=[names[i], names[j]], n_games=n_games, verbose=False)
                    mat[i, j] += c
                    mat[j, i] -= c

            # save data
            print()
            print(names)
            print(n)
            print(np.array(mat, dtype=int))
            with open(data_path, 'wb') as file:
                mat[0, 0] = n
                pickle.dump(mat, file)
                mat[0, 0] = 0

        # plot
        if do_plot:
            x = np.arange(len(names))
            fig, ax = plt.subplots(ncols=2)

            plt.sca(ax[0])
            plt.title(f'Tournament ({n} games)')
            plt.imshow(mat.T / n, cmap='bwr', origin='upper')
            lim = np.max(mat.flatten()) / n
            plt.clim(-lim, lim)
            for i in range(len(names)):
                for j in range(i):
                    p = (mat[i, j] + n+1) / (2*n+2)
                    e = 400 * np.log10(p/(1-p))
                    plt.text(j, i, f'{e:.0f}', c='w', size=13,
                             horizontalalignment='center', verticalalignment='center')
                    plt.text(i, j, f'{e:.0f}', c='w', size=13,
                             horizontalalignment='center', verticalalignment='center')
            plt.xticks(x, names, rotation=-30)
            plt.yticks(x, names)

            plt.sca(ax[1])
            y = np.average(mat, axis=1)
            plt.scatter(x, y)
            for i in range(len(names)):
                s = names[i].split(".")[0]
                s = s.replace('_', '\n')
                plt.text(x[i]+0.1, y[i], f'{s}', size=9, verticalalignment='center')
            plt.xlim(-0.5, len(names)+0.5)

            plt.show()

        if n_games == 0:
            break


if __name__ == '__main__':
    match(('value_function.pkl', 'softplus_100_100.pkl'))
    # tournament()
    # tournament(n_games=0, do_plot=True)
