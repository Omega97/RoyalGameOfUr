import numpy as np
from game.royal_game_of_ur import RoyalGameOfUr
from game.agent import Agent


def test_game():
    np.random.seed(0)
    game = RoyalGameOfUr()
    agents = [Agent(), Agent()]
    game.play(agents, verbose=True)


def test_rules(state_info):
    game = RoyalGameOfUr().set_state(state_info)
    print(game)
    legal = game.get_legal_moves()
    print(list(np.arange(len(legal))[legal > 0]))
    if not np.all(legal == state_info["legal_moves"]):
        raise ValueError(f'\n{game}\nlegal\n{legal}\ntrue legal\n{state_info["legal_moves"]}\n')


def test_legal_moves():
    state = {'current_player': 0,
             'n_steps': 2,
             'round': 81,
             'board': np.array([[3, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3]]),
             'board_size': 8,
             'n_dice': 4,
             'n_pieces': 7,
             'legal_moves': np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])}

    test_rules(state)


if __name__ == '__main__':
    test_game()
    # test_legal_moves()
