"""
The Royal Game of Ur is a two-player imperfect information game that dates back to ancient Mesopotamia.

action representation:
0: introduce a piece into the board
1 to 2*N-2: move a piece from that square on your side of the board
2*N-1: pass
"""
import numpy as np
import pickle
from copy import deepcopy
import os
from game import Game


def roll(n_dices):
    """roll n_dices 2-sided dices and return the sum of the results"""
    return np.random.randint(0, 2, n_dices).sum()


def index_to_coordinate(index, player, board_size, n_dice):
    """find the location of the square on the board"""
    assert 0 <= index < 2 * board_size
    length = n_dice + 1

    if index < length:
        row = 0
        col = n_dice - index
    elif index < board_size + length:
        row = 1
        col = index - length
    else:
        row = 0
        col = 2 * board_size + n_dice - index

    if player == 1:
        row = 2 - row
    return row, col


def board_to_screen(board, board_size, n_dice):
    """convert board to matrix representation"""
    screen = np.zeros([3, board_size], dtype=int)

    for p in range(2):
        for k in range(len(board[p])):
            value = board[p][k]
            if value > 0:
                if p:
                    value *= -1
                i, j = index_to_coordinate(k, p, board_size, n_dice)
                screen[i, j] = value

    return screen


class RoyalGameOfUr(Game):

    def __init__(self, board_size=8, n_dice=4, n_pieces=7):
        super().__init__()
        self.board_size = board_size
        self.starting_square = 0
        self.path_length = 2 * board_size
        self.score_square = self.path_length - 1      # last square
        self.pass_move_id = self.path_length - 1      # this move id corresponds to passing the turn
        self.n_dice = n_dice
        self.n_pieces = n_pieces
        self.special_positions = (n_dice, 2 * n_dice, self.score_square - 1)
        self.war_zone = np.zeros(self.path_length, dtype=int)
        self.war_zone[self.n_dice + 1: self.n_dice + self.board_size + 1] = 1  # war-zone

        self.pieces = ('O', 'X')

        self._do_update_player = None
        self.board = None
        self.current_player = None
        self.round = None
        self.n_steps = None  # by how many steps does the piece move

        self.game_record = None
        self.rolls_record = None
        self.player_moves = None
        self.evaluations = None
        self.turn_counter = None

        self.agents = None

        self.reset()

    def roll_dice(self):
        """
        Set the number of steps to move the piece.
        The roll is updated at the beginning of the game and after each move.
        """
        self.n_steps = roll(self.n_dice)

    def get_board(self):
        """get the board as a numpy array"""
        assert type(self.board) is not list
        return self.board

    def get_legal_moves(self):
        """
        Get the legal moves for the current player as a numpy binary array.

        Rules:
        - the player can move one of their pieces by the number of steps rolled
        - they can move a piece from the starting square by the same amount
        - they can capture an opponent's piece if it is on the destination square
        - they cannot move a piece on a square that is already occupied by one of their pieces
        - they cannot capture a piece on the special squares
        - if (and only if) they have no legal moves, they must pass the turn
        - to promote a piece to the score square, the player must roll the exact number of steps
        """
        legal = np.zeros(2 * self.board_size, dtype=int)
        opponent = 1 - self.current_player

        # your pieces can be moved by the number of steps rolled
        for start in range(self.path_length - self.n_steps):
            destination = start + self.n_steps

            # check if the square is occupied by your piece
            if self.board[self.current_player, start] > 0:
                legal[start] = 1

                # check if the destination square is occupied by your piece
                if self.board[self.current_player, destination] > 0:
                    if destination != self.score_square:
                        legal[start] = 0

                # check if the destination square is special (but only in the war-zone)
                if self.war_zone[destination]:
                    if destination in self.special_positions:
                        if self.board[opponent, destination] > 0:
                            legal[start] = 0

        # pass if there are no legal moves
        if legal.sum() == 0:
            legal[self.pass_move_id] = 1

        return legal

    def _move_piece(self, position, destination):
        """move a piece from a square to another square"""
        assert self.get_board()[self.current_player, position] > 0, f'No piece to move! \nmove={position}\n{self}\n'

        if destination in self.special_positions:
            self._do_update_player = False
        if destination != self.score_square:
            if self.board[self.current_player][destination] > 0:
                raise ValueError(f'Stone already present! \n{self}\n')
        self.board[self.current_player][position] -= 1
        self.board[self.current_player][destination] += 1

    def _capture_piece(self, destination):
        """capture a piece from the opponent"""
        if self.war_zone[destination] == 1:
            opp = 1 - self.current_player
            if self.board[opp][destination] > 0:
                self.board[opp][0] += self.board[opp][destination]
                self.board[opp][destination] = 0

    def move(self, position: int):
        """move a piece from a square to another square"""
        assert type(position) in (int, np.int32)
        self._do_update_player = True

        # move and capture
        if position != self.pass_move_id:
            destination = position + self.n_steps
            self._move_piece(position, destination)
            self._capture_piece(destination)

        # update player
        if self._do_update_player:
            self.current_player = 1 - self.current_player

    def reset(self):
        """player, x, y"""
        self.board = np.zeros([2, self.path_length], dtype=int)
        self.board[0, self.starting_square] = self.n_pieces
        self.board[1, self.starting_square] = self.n_pieces
        self.current_player = 0
        self.round = 0

        self.game_record = []
        self.rolls_record = []
        self.player_moves = []
        self.evaluations = []
        self.turn_counter = 0

        self.roll_dice()
        self._update_game_record()

    def _update_game_record(self):
        self.game_record.append(self.get_state_info())

    def is_game_over(self):
        """check if the game is over"""
        for player in range(2):
            if self.get_board()[player, self.score_square] == self.n_pieces:
                return True
        return False

    def get_reward(self) -> np.array:
        """get the reward for the current player (zeros if the game is not over)"""
        if self.get_board()[0, self.score_square] == self.n_pieces:
            # first player won
            return np.array([1., 0.])
        elif self.get_board()[1, self.score_square] == self.n_pieces:
            # second player won
            return np.array([0., 1.])
        else:
            # game not over yet
            return np.array([0., 0.])

    def __repr__(self):

        s = f'\n{self.round}) \t'
        s += f'player: {self.pieces[self.current_player]} \t'
        if not self.is_game_over():
            s += f'rolled: \033[92m{self.n_steps}\033[0m \t'
        s += '\n\n'

        mat = board_to_screen(self.board, self.board_size, self.n_dice)
        mat = mat.astype(str)

        # map special positions to characters
        pos = [0] + list(self.special_positions) + [self.path_length - 1]
        chars = '_' + '*' * len(self.special_positions) + '_'

        for p in range(2):
            for n, c in zip(pos, chars):
                i, j = index_to_coordinate(n, p, self.board_size, self.n_dice)
                if mat[i, j] == '0':
                    mat[i, j] = c

        mat[mat == '0'] = '.'
        mat[mat == '1'] = self.pieces[0]
        mat[mat == '-1'] = self.pieces[1]

        for v in mat:
            s += ' ' * 6
            for c in v:
                s += f'{c:>3}'
            s += '\n'

        # color character
        for c, n in [('O', 94), ('X', 91), ('_', 92), ('*', 93)]:
            s = s.replace(c, f'\033[{n}m{c}\033[0m')

        s = s.replace('-', ' ')

        return s

    def _get_agent_action(self, verbose):
        """get action from agent"""

        agent = self.agents[self.current_player]
        assert hasattr(agent, '__call__'), f'Agent {agent} is not callable'

        self.turn_counter += 1

        if verbose:
            print(self)

        # get action from agent
        state_info = self.get_state_info()
        agent_output = agent(state_info, verbose=verbose)

        assert type(agent_output) is dict, f'{agent} returned {agent_output}, should return a dict instead'
        assert "action" in agent_output
        assert "eval" in agent_output
        action = agent_output["action"]

        player_eval = agent_output["eval"]
        assert len(player_eval) == 2, f"please evaluate position for all players; {agent}(state)={player_eval}"

        if type(action) not in (int, np.int32):
            raise ValueError(f'type(action) = {type(action)}, should be int instead')

        return action, player_eval

    def _update_game_state(self, action, player_eval):

        # update game state
        self.round += 1
        self.move(action)

        # update roll
        self.roll_dice()

        # save game info
        self.game_record.append(self.get_state_info())
        self.rolls_record.append(self.n_steps)
        self.player_moves.append(action)
        self.evaluations.append(player_eval)

    def _get_game_recap(self) -> dict:
        """Returns the game recap as a dictionary
        """
        return {"players": (str(self.agents[0]), str(self.agents[1])),
                "reward": self.get_reward(),
                "game_record": self.game_record,
                "player_on_duty": tuple([s["current_player"] for s in self.game_record]),  # todo check
                "rolls": self.rolls_record,
                "player_moves": self.player_moves,
                "player_eval": self.evaluations,
                "is_game_over": self.is_game_over()
                }

    def _close_game(self, verbose=False):
        """close the game by adding final reward, then print the final state"""
        final_reward = self.get_reward()
        self.evaluations.append(final_reward)
        if verbose:
            print(self)
            winner = int(np.argmax(final_reward))
            s = f'\n\t\tPlayer {self.pieces[winner]} wins!\n'
            for c, n in [('O', 94), ('X', 91)]:
                s = s.replace(c, f'\033[{n}m{c}\033[0m')
            print(s)

    def play(self, agents, do_reset=True, max_depth=None, verbose=False) -> dict:
        """play the game until the end or until the maximum number of turns is reached
        :param agents: list of agents
        :param max_depth: maximum number of turns from the current state
        :param do_reset: reset the game before playing (if False, the game will continue from the current state)
        :param verbose: print the game state at each turn
        """

        # reset game state
        self.agents = agents
        for agents in self.agents:
            agents.reset()
        if do_reset:
            self.reset()

        while not self.is_game_over():
            # get action from agent
            action, player_eval = self._get_agent_action(verbose=verbose)

            # check legality of the move
            legal_moves = self.get_legal_moves()
            legal_indices = np.arange(len(legal_moves))[legal_moves > 0]
            assert action in legal_indices, f'Illegal move: {action} not in {legal_indices}\n{self}\n'

            # update game state
            self._update_game_state(action, player_eval)

            # check if the maximum number of turns has been reached
            if max_depth is not None:
                if self.turn_counter >= max_depth:
                    break

        # close the game
        if self.is_game_over():
            self._close_game(verbose)

        return self._get_game_recap()

    def save(self, path, verbose=False):
        """save the game recap as a pickle file"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            if verbose:
                print(f'Saving game to {path}')
            pickle.dump(self._get_game_recap(), f)

    def get_game_parameters(self) -> dict:
        """get game parameters as a dictionary"""
        return {"board_size": self.board_size,
                "n_dice": self.n_dice,
                "n_pieces": self.n_pieces,
                }

    def get_state_info(self) -> dict:
        """get a deepcopy of state info as a dictionary"""
        out = {"current_player": self.current_player,
               "n_steps": self.n_steps,
               "round": self.round,
               "board": self.board,
               "legal_moves": self.get_legal_moves(),
               }
        return deepcopy(out)

    def set_state(self, state_info):
        """set state info from a dictionary"""
        info_copy = deepcopy(state_info)  # without deepcopy, the original could be modified
        self.current_player = info_copy["current_player"]
        self.n_steps = info_copy["n_steps"]
        self.round = info_copy.get("round", 0)
        self.board = info_copy["board"]
        assert sum(info_copy["board"][0]) + sum(info_copy["board"][1]) == 2 * self.n_pieces
        return self

    def set_game_parameters(self, game_parameters):
        """set game parameters from a dictionary"""
        self.board_size = game_parameters.get("board_size", self.board_size)
        self.n_dice = game_parameters.get("n_dice", self.n_dice)
        self.n_pieces = game_parameters.get("n_pieces", self.n_pieces)
        return self

    def deepcopy(self):
        """return a deepcopy of the game"""
        return RoyalGameOfUr().set_state(self.get_state_info())
