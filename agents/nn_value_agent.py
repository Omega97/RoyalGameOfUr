import os.path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from game.evaluation import evaluation_match
from game.agent import Agent
from agents.random_agent import RandomAgent
from game.training_data import state_to_features
from src.utils import bar, bcolors, cprint


class NNValueAgent(Agent):

    PROBA = np.array([1, 4, 6, 4, 1]) / 16

    def __init__(self,
                 game_instance,
                 models_dir_path,
                 value_function_builder=None,
                 value_function_name="value_function.pkl",
                 reward_half_life=10,
                 verbose=False):

        self.game = game_instance
        self.dir_path = models_dir_path
        self.value_function_builder = value_function_builder
        self.value_function_name = value_function_name
        self.gamma = np.log(2) / reward_half_life
        self.game_input_size = game_instance.INPUT_SIZE

        self.value_function = None
        self.legal_indices = None
        self.scores = None
        self.states = None

        self.reset(verbose=verbose)

    def get_value_function(self):
        return self.value_function

    def reset_value_function(self, input_size, output_size=2):
        """ MLP value function """
        cprint('Resetting value function', bcolors.WARNING)
        assert self.value_function_builder is not None, "Value function builder is not set"
        self.value_function = self.value_function_builder(input_size, output_size)

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
        return os.path.join(self.dir_path, self.value_function_name)

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

    def train_value_function(self, x, y_target,
                             lr=2e-5,
                             n_epoch=100,
                             verbose=True):
        """Train the value function using the training data"""

        # Define loss function
        criterion = nn.modules.loss.MSELoss()
        # criterion = nn.modules.loss.BCELoss()

        # Define optimizer
        # optimizer = torch.optim.SGD(self.value_function.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam(self.value_function.parameters(), lr=lr)
        # optimizer = torch.optim.Adagrad(self.value_function.parameters(), lr=lr)

        # metrics before training
        loss_before = 0.
        rmse_before = 0.
        if verbose:
            y_model_before = self.value_function(x.float())
            loss_before = criterion(y_model_before, y_target)
            y_model_before = y_model_before.detach().numpy()
            rmse_before = np.sqrt(np.mean((y_target[:, 0] - y_model_before[:, 0]).numpy() ** 2))

        # update
        for epoch in range(n_epoch):
            if verbose:
                if n_epoch > 1:
                    if (epoch+1) % 10 == 0:
                        print(f'\rEpoch {epoch + 1}/{n_epoch}', end='')
            y_model = self.value_function(x.float())
            loss = criterion(y_model, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            if n_epoch > 1:
                print()

        # metrics after training
        if verbose:
            y_model_after = self.value_function(x.float())
            loss_after = criterion(y_model_after, y_target)
            y_model_after = y_model_after.detach().numpy()
            rmse_after = np.sqrt(np.mean((y_target[:, 0] - y_model_after[:, 0]).numpy() ** 2))

            # print data and results
            p = (rmse_after - rmse_before) / rmse_before
            message = f'RMSE: {rmse_before:.5f} -> {rmse_after:.5f}  ({p:+.2%})'
            cprint(message, bcolors.OKGREEN if p < 0 else bcolors.FAIL)
            p = (loss_after - loss_before) / loss_before
            message = f'Loss: {loss_before:.5f} -> {loss_after:.5f}  ({p:+.2%})'
            cprint(message, bcolors.OKGREEN if p < 0 else bcolors.FAIL)

    def train_agent(self, x, y_policy, y_value, verbose=True):
        """Train the value function using the training data"""
        # check for value function
        if self.value_function is None:
            raise ValueError("Value function is not set")
        elif verbose:
            cprint(f"Training value function on {len(x)} data-points", bcolors.CYAN)

        # run training
        self.train_value_function(x, y_value, verbose=verbose)

        # save value function
        cprint(f'Saving value function to {self.get_value_function_path()}')
        torch.save(self.value_function, self.get_value_function_path())
        cprint(f'Value function saved', bcolors.OKGREEN)

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
