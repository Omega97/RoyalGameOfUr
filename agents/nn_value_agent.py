import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from game.evaluation import evaluation_match
from game.agent import Agent
from agents.random_agent import RandomAgent
from game.training_data import state_to_features
from src.utils import bar, bcolors, cprint


def initialize_weights(model, weight_range=0.1):
    """
    Initialize weights and biases
    of the model within the given range
    """
    assert weight_range > 0, "Weight range must be positive"
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)


class NNValueAgent(Agent):

    PROBA = np.array([1, 4, 6, 4, 1]) / 16

    def __init__(self,
                 game_instance,
                 models_dir_path,
                 depth=2,
                 reward_half_life=10,
                 verbose=False):

        self.game = game_instance
        self.dir_path = models_dir_path
        self.depth = depth
        self.gamma = np.log(2) / reward_half_life
        self.game_input_size = game_instance.INPUT_SIZE

        self.value_function = None
        self.legal_indices = None
        self.scores = None
        self.states = None

        self.reset(verbose=verbose)

    def __repr__(self):
        return f"NewNNValueAgent(depth={self.depth})"

    def get_value_function(self):
        return self.value_function

    def reset_value_function(self, input_size, hidden_units=160,
                             weight_range=0.1, output_size=2):
        """ MLP value function """
        cprint('Resetting value function', bcolors.WARNING)

        # architecture 1
        # self.value_function = (
        #     nn.Sequential(
        #         nn.Linear(input_size, hidden_units),
        #         nn.ReLU(),
        #         nn.Linear(hidden_units, output_size),
        #         nn.Sigmoid()
        #     ))

        # architecture 2
        self.value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Softplus(),
                nn.Linear(hidden_units, output_size),
                nn.Sigmoid()
            ))

        initialize_weights(self.value_function, weight_range)

    def evaluate_states_after_roll(self, state_info_list: List[Dict]):
        """
        Evaluate multiple states using the value function
        """
        features = [state_to_features(state_info) for state_info in state_info_list]
        x = np.array(features)
        x = torch.tensor(x, dtype=torch.float)
        return self.value_function(x).detach().numpy()

    def reset(self, verbose=False):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        try:
            # cprint(f'\nTrying to load model from {self.get_value_function_path()}...', bcolors.WARNING)
            self.value_function = torch.load(self.get_value_function_path())
            # cprint('Model loaded successfully!', bcolors.OKGREEN)
        except FileNotFoundError:
            cprint('\nLoading failed. Creating new models...', bcolors.WARNING)
            self.reset_value_function(input_size=self.game_input_size)
            torch.save(self.value_function, self.get_value_function_path())
            cprint('\nValue function created', bcolors.OKGREEN)

    def get_value_function_path(self):
        return os.path.join(self.dir_path, "value_function.pkl")

    def _evaluate_state_before_roll(self, state, player_id):
        """
        Calculate the expected value of a state before rolling the dice
        by averaging the values of the states after all possible rolls
        """
        values = np.zeros(len(NNValueAgent.PROBA))
        indices = np.array([], dtype=int)
        info = []

        # iterate over all rolls
        for j in range(len(values)):
            state_2 = state.deepcopy()
            state_2.n_steps = j             # set the roll
            state_2_info = state_2.get_state_info()

            indices = np.append(indices, j)
            info.append(state_2_info)

            if state_2.is_game_over():
                values[j] = state_2.get_reward()[player_id]

        # evaluate all states in bulk
        if len(info):
            y = self.evaluate_states_after_roll(info)
            y = y[:, player_id]
            values[indices] = y

        return np.dot(values, NNValueAgent.PROBA)

    def _get_move_values(self, legal_move_indices, state_info, player_id):
        """Get the value of each move in legal_moves"""
        values = np.zeros(len(legal_move_indices))

        # iterate over all legal moves
        for i in range(len(legal_move_indices)):
            state = self.game.deepcopy()
            state = state.set_state(state_info)
            move = legal_move_indices[i]

            # get new state after move
            state.move(move)

            # evaluate state
            if state.is_game_over():
                value = state.get_reward()[player_id]
            else:
                value = self._evaluate_state_before_roll(state, player_id)
            values[i] = value

        return values

    def _get_best_move(self, state_info):
        """
        Get the best move and its expected value
        after the dice roll
        """

        # get move values
        player_id = state_info["current_player"]
        legal_moves = state_info["legal_moves"]
        legal_move_indices = np.arange(len(legal_moves))[legal_moves > 0]
        values = self._get_move_values(legal_move_indices, state_info, player_id)

        # pick move with best value
        best_i = np.argmax(values)
        best_move = legal_move_indices[best_i]
        best_eval = values[best_i]
        expected_reward = [best_eval, 1 - best_eval]
        if player_id == 1:
            expected_reward = list(reversed(expected_reward))
        expected_reward = np.array(expected_reward)

        return best_move, expected_reward

    def get_action(self, state_info: dict, verbose=False, bar_length=30, half_life=10) -> dict:
        """
        Evaluate all the possible moves and return
        the one with the highest expected value.
        """
        assert self.value_function is not None
        player_id = state_info["current_player"]

        # get best move
        best_move, expected_reward = self._get_best_move(state_info)
        best_eval = expected_reward[player_id]

        if verbose:
            print(f'action {best_move}')
            color = bcolors.RED if player_id == 1 else bcolors.BLUE
            print(f'{bar(p=best_eval, color=color, length=bar_length)} {best_eval:.3f}')

        return {"action": best_move, "eval": expected_reward}

    def print_search_results(self, state_info, ev, t, bar_length):
        """Print the results of the search"""
        color = bcolors.RED if state_info["current_player"] == 1 else bcolors.BLUE
        best_value_index = int(np.argmax(ev))

        cprint('\nSearch EV:')
        for i in range(len(self.legal_indices)):
            p = ev[i]
            s = f'{self.legal_indices[i]:3})  {p:6.2%}  '
            s += bar(p, color, length=bar_length)
            if i == best_value_index:
                s += ' <<'
            print(s)

        print(f"\nTime: {t:.2f} s")

    def train_value_function(self, x, y_target, lr=0.1,
                             momentum=0.9, n_epoch=1, verbose=True):
        """Train the value function using the training data"""
        if self.value_function is None:
            raise ValueError("Value function is not set")
        elif verbose:
            cprint(f"Training value function on {len(x)} data-points", bcolors.CYAN)

        # Define loss function
        criterion = nn.MSELoss()
        # criterion = nn.modules.loss.BCELoss()

        # Define optimizer
        # optimizer = optim.SGD(self.value_function.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(self.value_function.parameters(), lr=lr)
        # optimizer = optim.Adagrad(self.value_function.parameters(), lr=lr)

        y_model_before = self.value_function(x.float())
        loss_before = criterion(y_model_before, y_target)
        y_model_before = y_model_before.detach().numpy()
        rmse_before = np.sqrt(np.mean((y_target[:, 0] - y_model_before[:, 0]).numpy() ** 2))

        # update
        for epoch in range(n_epoch):
            # Forward pass
            y_model = self.value_function(x.float())
            loss = criterion(y_model, y_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimize
            optimizer.step()

        # print data and results
        y_model_after = self.value_function(x.float()).detach().numpy()
        got_better = np.array(np.abs(y_target[:, 0] - y_model_before[:, 0]) > abs(y_target[:, 0] - y_model_after[:, 0]))
        p_got_better = np.mean(got_better.flatten())
        cprint(f'Got better: {p_got_better:.1%}', bcolors.WARNING if p_got_better < 0.5 else bcolors.OKGREEN)

        rmse_after = np.sqrt(np.mean((y_target[:, 0] - y_model_after[:, 0]).numpy() ** 2))
        p = (rmse_after - rmse_before) / rmse_before
        cprint(f'RMSE: {rmse_before:.5f} -> {rmse_after:.5f}  ({p:+.2%})', bcolors.OKGREEN if p < 0 else bcolors.FAIL)

        # for i in range(len(x)):
        #     for j in range(len(x[i])):
        #         if x[i][j] == 0:
        #             print('_', end='')
        #         else:
        #             print('#', end='')
        #     print(f'  {y_target[i][0]:7.2%} | {y_model[i][0]:6.2%} -> ', end='')
        #     cprint(f'{y_model_after[i][0]:6.2%}', bcolors.OKGREEN if got_better[i] else bcolors.FAIL)
        # input('>>>')

        output_after = self.value_function(x.float())
        loss_after = criterion(output_after, y_target)
        p = (loss_after - loss_before) / loss_before
        cprint(f'Loss: {loss_before:.5f} -> {loss_after:.5f}  ({p:+.2%})', bcolors.OKGREEN if p < 0 else bcolors.FAIL)

        # save value function
        cprint(f'Saving value function to {self.get_value_function_path()}')
        torch.save(self.value_function, self.get_value_function_path())
        cprint(f'Value function saved', bcolors.OKGREEN)

    def train_agent(self, x, y_policy, y_value, lr=0.1, verbose=True):
        """Train the value function using the training data"""
        return self.train_value_function(x, y_value, lr=lr, verbose=verbose)

    def evaluate(self, n_games=50, show_game=False):
        """Evaluate the components of the agent (value function)"""
        print(f'\nEvaluating agent on {n_games} games...')

        # agents
        random_agent = RandomAgent()

        elo = 0
        for order in range(2):
            agents = [self, random_agent]
            if order == 1:
                agents = list(reversed(agents))
            print(f'\nAgent 1: {agents[0]}\nAgent 2: {agents[1]}\n')

            # evaluation match
            elo += evaluation_match(*agents, n_games=n_games, show_game=show_game, player=order)

        return elo / 2
