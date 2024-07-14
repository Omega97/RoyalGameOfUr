import numpy as np
import pickle
import torch


def one_hot_encode(n, size):
    """ One hot encoding of n in a vector of given size """
    v = [0] * size
    v[n] = 1
    return v


def state_to_features(state_info, default_n_dice=4, default_n_pieces=7):
    """
    Convert state info to array (abstract board representation)
    # ['current_player', 'n_steps', 'board', 'board_size', 'legal_moves']
    """
    v = []
    n_dice = state_info.get("n_dice", default_n_dice)
    n_pieces = state_info.get("n_pieces", default_n_pieces)

    # player id (0:1)
    v.append(state_info["current_player"])

    # n_steps (1:6)
    v.extend(one_hot_encode(state_info["n_steps"], n_dice + 1))

    # pieces in the inventory (6:14) (14:22)
    v.extend(one_hot_encode(state_info["board"][0][0], n_pieces+1))
    v.extend(one_hot_encode(state_info["board"][1][0], n_pieces+1))

    # pieces on the board (22:36) (36:50)
    v.extend(state_info["board"][0][1:-1])
    v.extend(state_info["board"][1][1:-1])

    # promoted pieces (50:58) (58:66)
    v.extend(one_hot_encode(state_info["board"][0][-1], n_pieces+1))
    v.extend(one_hot_encode(state_info["board"][1][-1], n_pieces+1))

    # legal moves (67:83)
    v.extend(state_info["legal_moves"])

    return np.array(v)


def output_to_action(output):
    """ convert agent's output to action """
    return int(np.argmax(output))


def backpropagation(v, halflife=20):
    """
    Propagate the value back in time
    v: list of values
    halflife: signal decay time
    """
    v = np.array(v)
    if halflife > 0:
        k = np.exp(-np.log(2) / halflife)  # 2^(-1/x)
    else:
        k = 0.
    out = np.zeros_like(v)
    out[-1] = v[-1]
    for i in reversed(range(len(v)-1)):
        out[i] = out[i+1] * k + v[i] * (1-k)
    return out


def create_dataset_from_game_files(game_files, halflife=10):
    """
    Build the dataset, then return X, y_policy, and y_value as numpy arrays
    """
    x = []
    y_policy = []
    y_value = []

    for file in game_files:
        dct = pickle.load(open(file, 'rb'))
        y_value_new = []

        for i in range(len(dct['game_record'])-1):
            state = dct['game_record'][i]

            # input features
            x.append(state_to_features(state))

            # move
            y_policy.append(one_hot_encode(dct['player_moves'][i], len(state['legal_moves'])))

            # value
            y_value_new.append(dct['player_eval'][i])

        y_value_new.append(dct["reward"])
        y_value_new = backpropagation(y_value_new, halflife)
        y_value_new = y_value_new[:-1]

        y_value.extend(y_value_new)

    x = np.array(x)
    y_policy = np.array(y_policy)
    y_value = np.array(y_value)

    x = torch.tensor(x, dtype=torch.float)
    y_policy = torch.tensor(y_policy, dtype=torch.float)
    y_value = torch.tensor(y_value, dtype=torch.float)

    return x, y_policy, y_value


def multiple_one_hot(x: np.array, n_values: np.array, dtype=np.float64):  # todo use in sate_to_features
    """create the one-hot vector with multiple distributions"""

    print(x)
    print(n_values)

    assert len(x) == len(n_values)
    w = np.cumsum(np.concatenate((np.zeros(1, dtype=np.int64), n_values[:-1])))
    v = np.zeros(sum(n_values), dtype)
    v[w + x] += 1
    return v


def _state_to_features(state_info, default_n_dice=4, default_n_pieces=7, board_length=8):
    """
    Convert state info to array (abstract board representation)
    # ['current_player', 'n_steps', 'board', 'board_size', 'legal_moves']
    # todo fix pieces on the board
    """
    n_dice = state_info.get("n_dice", default_n_dice)
    n_pieces = state_info.get("n_pieces", default_n_pieces)
    n_squares = board_length * 2 - 1

    x = (state_info["current_player"],  # player id
         state_info["n_steps"],                        # roll
         state_info["board"][0][0],     # pieces in the inventory
         state_info["board"][1][0],
         state_info["board"][0][1:-1],          # pieces on the board
         state_info["board"][1][1:-1],
         state_info["board"][0][-1],         # promoted pieces
         state_info["board"][1][-1]
         )

    n_values = (2,                          # player id
                n_dice + 1,                 # roll
                n_pieces + 1,               # pieces in the inventory
                n_pieces + 1,
                n_squares,                  # pieces on the board
                n_squares,
                n_pieces + 1,               # promoted pieces
                n_pieces + 1
                )

    v = multiple_one_hot(np.array(x), np.array(n_values))

    v = np.concatenate(v, np.array(state_info["legal_moves"]))

    return v
