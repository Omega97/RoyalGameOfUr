"""
Most common starting positions (proba and rolls)
...
0.88%   (2, 2, 4) (2, 4, 2)
0.89%   (2, 1, 2, 1) (2, 1, 2, 3) (2, 3, 2, 1) (2, 3, 2, 3)
1.34%   (2, 1, 2, 2) (2, 2, 2, 1) (2, 2, 2, 3) (2, 3, 2, 2)
1.57%   (4, 1) (4, 3)
1.58%   (1, 1, 3) (1, 3, 3) (3, 1, 1) (3, 3, 1)
2.01%   (2, 2, 2, 2)
2.36%   (4, 2)
2.37%   (1, 1, 2) (1, 2, 3) (1, 3, 2) (2, 1, 1) (2, 1, 3)
        (2, 3, 1) (2, 3, 3) (3, 1, 2) (3, 2, 1) (3, 3, 2)
3.56%   (1, 2, 2) (2, 2, 1) (2, 2, 3) (3, 2, 2)
"""
import numpy as np
from agents.random_agent import RandomAgent
from game.royal_game_of_ur import RoyalGameOfUr


def play_random_game():
    game = RoyalGameOfUr()
    game_recap = game.play(agents=(RandomAgent(), RandomAgent()), verbose=False)

    rolls = game_recap["rolls"]
    n_legal_moves = game_recap["n_legal_moves"]

    for i in range(len(n_legal_moves)):
        if n_legal_moves[i] > 1:
            return tuple(rolls[:i+1])


def compute_proba(v):

    p = np.array([1, 4, 6, 4, 1]) / 16
    out = 1
    for j in v:
        out *= p[j]

    # correct for zeros
    out *= (1+len(v)/16**2)

    return out


def main(n=4000, period=10):

    s = set()
    for i in range(n):
        if i % period == 0:
            print(f'\r{i}/{n}', end='')
        s.add(play_random_game())
    print()

    d = dict()
    for t in s:
        d[t] = compute_proba(t)

    new_d = dict()

    for t, p in d.items():
        k = f'{p:.3%}'
        new_d[k] = new_d.get(k, []) + [t]

    for k in sorted(new_d):
        print(k, ' ', ' '.join([str(t) for t in sorted(new_d[k])]))


if __name__ == '__main__':
    main()

