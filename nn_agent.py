import os.path

import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from training import Training
from royal_game_of_ur import RoyalGameOfUr
from agent import Agent
from training_data import state_to_features


def initialize_weights(model, weight_range=0.1):
    """ Initialize weights within the given range """
    assert weight_range > 0, "Weight range must be positive"
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)


class PolicyAgent(Agent):
    """This bot uses the policy to make decisions"""

    def __init__(self, policy=None, greedy=False, policy_path=None):
        self.policy = policy
        self.greedy = greedy
        self.policy_path = policy_path

        # set policy
        if self.policy is None:
            self.reset()

    def reset(self):
        self._load_from_path()

    def _load_from_path(self):
        """Load policy from path"""
        if self.policy_path is not None:
            assert os.path.exists(self.policy_path), 'File not found'
            self.policy = torch.load(self.policy_path)
            assert self.policy is not None, 'Failed to load policy'

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """Sample a random action according to the
        probability distribution of the policy"""
        assert self.policy is not None
        features = state_to_features(state_info)
        X = np.array([features])
        X = torch.tensor(X, dtype=torch.float)
        y = self.policy(X).detach().numpy()
        y *= state_info["legal_moves"]
        assert np.sum(y) > 0, "No legal moves"

        if self.greedy:
            action = int(np.argmax(y[0]))
        else:
            action = np.random.choice(len(y[0]), p=y[0] / np.sum(y[0]))  # todo normalization?

        return {"action": action}


class NNAgent(Agent):

    def __init__(self,
                 game_instance,
                 models_dir,
                 n_rollouts=100,
                 rollout_depth=5):
        self.game = game_instance
        self.dir_path = models_dir
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth

        self.policy = None
        self.value_function = None
        self.policy_agent = None
        self.legal_indices = None
        self.visits = None
        self.scores = None
        self.states = None

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self.reset()

    def __repr__(self):
        return f"NNAgent(n_rollouts={self.n_rollouts}, rollout_depth={self.rollout_depth})"

    def reset_policy(self, input_size, hidden_units, output_size):
        """ MLP Policy """
        self.policy = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Sigmoid(),
                nn.Linear(hidden_units, output_size),
                nn.Softmax(dim=1)
            ))

    def reset_value_function(self, input_size, hidden_units, output_size, weight_range=0.01):
        """ MLP value function """
        self.value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Sigmoid(),
                nn.Linear(hidden_units, output_size),
                nn.Sigmoid()
            ))
        initialize_weights(self.value_function, weight_range)

    def _set_policy_and_value_function(self, input_size=82, hidden_units=16, n_moves=16, verbose=False):
        """Try loading policy.pkl and value_function.pkl from dir_path, otherwise create new ones"""
        try:
            if verbose:
                print(f'\n>>> Trying to load models from {self.dir_path}')
            policy_path = f"{self.dir_path}\\policy.pkl"
            value_function = f"{self.dir_path}\\value_function.pkl"
            self.policy = torch.load(policy_path)
            self.value_function = torch.load(value_function)
            if verbose:
                print('>>> Models loaded successfully!')
        except FileNotFoundError:
            if verbose:
                print('\n>>> Loading failed. Creating new models...')
            self.reset_policy(input_size=input_size, hidden_units=hidden_units, output_size=n_moves)
            self.reset_value_function(input_size=input_size, hidden_units=hidden_units, output_size=2)

    def _set_policy_agent(self):
        """Set the policy agent using the policy"""
        self.policy_agent = PolicyAgent(self.policy)

    def reset(self):
        self._set_policy_and_value_function()
        self._set_policy_agent()

    def _reset_search(self, state_info):
        self.legal_indices = np.arange(len(state_info["legal_moves"]))[state_info["legal_moves"] > 0]
        self.n_moves = len(self.legal_indices)

        if self.n_moves > 1:
            self.visits = np.zeros(self.n_moves)
            self.scores = np.zeros(self.n_moves)
            self.states = []

            # create a list of states after each move
            for i in range(self.n_moves):
                state = self.game.deepcopy()  # without deepcopy, this doesn't work
                state = state.set_state(state_info)
                state.move(self.legal_indices[i])
                self.states.append(state)

    def evaluate_state(self, state_info: dict):
        """
        Evaluate a state using the value function
        """
        features = state_to_features(state_info)
        X = np.array([features])
        X = torch.tensor(X, dtype=torch.float)
        y = self.value_function(X).detach().numpy()
        return y[0]

    def call_policy(self, state_info: dict):
        """
        Call the policy in the state
        """
        features = state_to_features(state_info)
        X = np.array([features])
        X = torch.tensor(X, dtype=torch.float)
        y = self.policy(X).detach().numpy()
        return y[0]

    def _get_random_rollout_depth(self):
        """
        Get a random number between 1 and self.rollout_depth, or
        None if self.rollout_depth is None.
        """
        return self.rollout_depth  # todo implement

    def _rollout(self, state: RoyalGameOfUr, player_id):
        """
        Rollout a state for rollout_depth moves
        """

        assert self.policy_agent is not None, "Policy agent is not set"
        game_recap = state.play(agents=[self.policy_agent, self.policy_agent],
                                do_reset=False,
                                max_depth=self._get_random_rollout_depth(),
                                verbose=False)

        reward = game_recap["reward"]
        state_info = state.get_state_info()
        if reward is None:
            reward = self.evaluate_state(state_info)

        # print(reward)

        return reward[player_id]

    def _visit(self, move_index, player_id):
        """Update the visit and score arrays for a given move index"""
        state = self.states[move_index].deepcopy()  # without deepcopy, this doesn't work
        value = self._rollout(state, player_id)
        self.visits[move_index] += 1
        self.scores[move_index] += value

    def _get_output(self, state_info: dict):
        """After all the process to get the action and state evaluation
        is completed, return the action and evaluation """
        values = self.scores / self.visits
        best_index = int(np.argmax(values))
        evaluation = values[best_index]
        evaluation = [evaluation, 1 - evaluation] if state_info["current_player"] == 0 else [1 - evaluation, evaluation]
        action = self.legal_indices[best_index]
        return {"action": action, "eval": evaluation}

    def get_action(self, state_info: dict, verbose=False, bar_length=30) -> dict:
        """
        For every possible move, the agent makes n_rollouts rollouts
        and evaluates the position after rollout_depth moves using
        the value function. Return the move with the highest expected value.
        """
        # todo run simulations in parallel
        t = time()
        self._reset_search(state_info)

        # only legal move
        if self.n_moves == 1:
            return {"action": self.legal_indices[0]}

        # rollouts
        player_id = state_info["current_player"]
        for i in range(self.n_moves):
            for _ in range(self.n_rollouts):
                self._visit(i, player_id=player_id)

        # get index of best move
        ev = self.scores / self.visits
        best_index = int(np.argmax(self.scores / self.visits))

        # print
        if verbose:
            policy = self.call_policy(state_info)
            value = self.evaluate_state(state_info)
            print(f'\n>>> {self.n_rollouts} rollouts, depth {self.rollout_depth}')
            print(f'value: {value[player_id]:.3f}')
            print()

            for i in range(self.n_moves):
                color = 91 if player_id == 1 else 94

                p_2 = policy[self.legal_indices[i]]
                s_2 = f'{self.legal_indices[i]:3}: {p_2:6.1%}  |'
                s_2 += f'\033[{color}m' + '.' * int(p_2 * bar_length) + '\033[0m'
                print(s_2)

                p = ev[i]
                s = ' ' * 7 + f'{p:.3f} '
                s += f'[\033[{color}m' + '=' * int(p * bar_length) + '\033[0m]'
                if i == best_index:
                    s += ' <<'
                print(s)

            # print(f'Best move: {self.legal_indices[best_index]}')
            print()
            t = time() - t
            print(f"Time: {t:.2f}s")

        return self._get_output(state_info)

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function

    def _get_policy_accuracy(self, x, y_true):
        """ example tensors -> acc"""
        y = y_true.detach().numpy()
        y_true_values = np.argmax(y, axis=1)
        y_pred = self.policy(x)
        y_pred_values = np.argmax(y_pred.detach().numpy(), axis=1)
        accuracy = np.mean(y_pred_values == y_true_values)
        return accuracy

    def _get_value_function_rmse(self, x, y_true):
        """ example tensors -> rmse"""
        y = y_true.detach().numpy()
        y_pred = self.value_function(x)
        rmse = np.sqrt(np.mean((y_pred.detach().numpy() - y) ** 2))
        return rmse

    def train_policy(self, x, y,
                     n_epochs=500, lr=0.1,
                     momentum=0.9, batch_size=100,
                     print_period=50, verbose=True):
        """Train the policy using the training data"""
        if self.policy is None:
            raise ValueError("Policy is not set")
        if verbose:
            print("\n>>> Training policy")

        # stochastic gradient descent with momentum and fixed batch size
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.policy.parameters(), lr=lr, momentum=momentum)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []

        # accuracy
        if verbose:
            acc = self._get_policy_accuracy(x, y)
            print(f' accuracy: {acc:.1%}')

        for epoch in range(n_epochs):
            if verbose and (epoch + 1) % print_period == 0:
                losses = []

            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()
                output = self.policy(x_batch.float())
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                if verbose and (epoch + 1) % print_period == 0:
                    losses.append(loss.item())

            # performance
            if verbose and (epoch + 1) % print_period == 0:
                print(f'Epoch {epoch + 1:5}, Loss: {np.average(losses):.5f}')

        # accuracy
        if verbose:
            acc = self._get_policy_accuracy(x, y)
            print(f' accuracy: {acc:.1%}')

    def train_value_function(self, x, y,
                             n_epochs=1000, lr=0.1,
                             momentum=0.9, batch_size=100,
                             print_period=50, verbose=True):
        """Train the value function using the training data"""
        if self.value_function is None:
            raise ValueError("Value function is not set")
        if verbose:
            print("\n>>> Training value function")

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.value_function.parameters(), lr=lr, momentum=momentum)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []

        # rmse
        if verbose:
            rmse = self._get_value_function_rmse(x, y)
            print(f'     rmse: {rmse:.4f}')

        for epoch in range(n_epochs):
            if verbose and (epoch + 1) % print_period == 0:
                losses = []

            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()
                output = self.value_function(x_batch.float())
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                if verbose and (epoch + 1) % print_period == 0:
                    losses.append(loss.item())

            # performance
            if verbose and (epoch + 1) % print_period == 0:
                print(f'Epoch {epoch + 1:5}, Loss: {np.average(losses):.5f}')

        # rmse
        if verbose:
            rmse = self._get_value_function_rmse(x, y)
            print(f'     rmse: {rmse:.4f}')

    def train_agent(self, x, y_policy, y_value,
                    n_epochs_policy=500, n_epochs_value=500,
                    lr=0.1, momentum=0.9, verbose=True):
        """Train the policy and value function using the training data"""
        self.train_policy(x, y_policy, n_epochs=n_epochs_policy, lr=lr, momentum=momentum, verbose=verbose)
        self.train_value_function(x, y_value, n_epochs=n_epochs_value, lr=lr, momentum=momentum, verbose=verbose)

    def save_models(self, verbose=True):
        """Save the policy and value function to dir_path"""
        if verbose:
            print(f"\n>>> Saving models in {self.dir_path}")
        torch.save(self.policy, f"{self.dir_path}\\policy.pkl")
        torch.save(self.value_function, f"{self.dir_path}\\value_function.pkl")
