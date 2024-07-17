import numpy as np
from time import time
from game.royal_game_of_ur import RoyalGameOfUr


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
    if verbose:
        print()

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
            print(f'\rscore: {out[player]:3.0f} / {i + 1:3}   '
                  f'elo: {e[player]:.0f}   '
                  f'eta: {eta // 60:.0f} min {eta % 60:.0f} s', end='  ')

    if verbose:
        print()
    return score
