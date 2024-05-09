import os.path

import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

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


def create_policy(input_size, hidden_units, output_size, weight_range=0.01):
    """ MLP Policy """
    policy = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.Sigmoid(),
        nn.Linear(hidden_units, output_size),
        nn.Softmax(dim=1)
    )

    initialize_weights(policy, weight_range)

    return policy


def create_value_function(input_size, hidden_units, output_size, weight_range=0.01):
    """ MLP value function """
    value_function = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.Sigmoid(),
        nn.Linear(hidden_units, output_size),
        nn.Sigmoid()
    )

    initialize_weights(value_function, weight_range)

    return value_function


class PolicyAgent(Agent):
    """This bot uses the policy to make decisions"""

    def __init__(self, policy):
        self.policy = policy

    def get_action(self, state_info: dict, verbose=False) -> dict:
        """Get the action with the highest probability from the policy"""
        features = state_to_features(state_info)
        X = np.array([features])
        X = torch.tensor(X, dtype=torch.float)
        y = self.policy(X).detach().numpy()
        y *= state_info["legal_moves"]
        action = int(np.argmax(y[0]))
        return {"action": action}


class NNAgent(Agent):

    def __init__(self,
                 game_instance,
                 models_dir,
                 n_rollouts=100,
                 rollout_depth=20):
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

    def _set_policy_and_value_function(self, input_size=82, hidden_units=100, n_moves=16, verbose=True):
        """Try loading policy.pkl and value_function.pkl from dir_path, otherwise create new ones"""
        try:
            if verbose:
                print(f'\n>>> Trying to load models from {self.dir_path}')
            policy_path = f"{self.dir_path}\\policy.pkl"
            value_function = f"{self.dir_path}\\value_function.pkl"
            self.policy = torch.load(policy_path)
            self.value_function = torch.load(value_function)
            print('>>> Models loaded successfully!')
        except FileNotFoundError:
            print('\n>>> Loading failed; creating new models...')
            self.policy = create_policy(input_size=input_size, hidden_units=hidden_units, output_size=n_moves)
            self.value_function = create_value_function(input_size=input_size, hidden_units=hidden_units, output_size=2)

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
                state = self.game.deepcopy()             # without deepcopy, this doesn't work
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
        return self.rollout_depth   # todo implement

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
        state = self.states[move_index].deepcopy()       # without deepcopy, this doesn't work
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
        the value function. The move with the highest expected value
        is chosen.
        """
        t = time()
        self._reset_search(state_info)

        # only legal move
        if self.n_moves == 1:
            return {"action": self.legal_indices[0]}

        # rollouts
        for i in range(self.n_moves):

            # print(f'\n Move {self.legal_indices[i]}')

            for _ in range(self.n_rollouts):
                self._visit(i, player_id=state_info["current_player"])

        # get index of best move
        ev = self.scores / self.visits
        best_index = int(np.argmax(self.scores / self.visits))

        # print
        if verbose:
            policy = self.call_policy(state_info)
            for i in range(self.n_moves):
                color = 91 if state_info["current_player"] == 1 else 94

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

    def _train_policy(self, x, y,
                      n_epochs=1000, lr=0.1,
                      momentum=0.9, verbose=True):
        """Train the policy using the training data"""
        if self.policy is None:
            raise ValueError("Policy is not set")
        if verbose:
            print("\n>>> Training policy")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.policy.parameters(),
                              lr=lr, momentum=momentum)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.policy(x.float())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # performance
            if verbose and (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1:5}, Loss: {loss.item():.5f}')

    def _train_value_function(self, x, y, n_epochs=1000, lr=0.1, momentum=0.9, verbose=True):
        """Train the value function using the training data"""
        if self.value_function is None:
            raise ValueError("Value function is not set")
        if verbose:
            print("\n>>> Training value function")

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.value_function.parameters(), lr=lr, momentum=momentum)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.value_function(x.float())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # performance
            if verbose and (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1:5}, Loss: {loss.item():.5f}')

    def train_agent(self, x, y_policy, y_value, n_epochs=4000, lr=0.1, momentum=0.9, verbose=True):
        """Train the policy and value function using the training data"""
        self._train_policy(x, y_policy, n_epochs=n_epochs, lr=lr, momentum=momentum, verbose=verbose)
        self._train_value_function(x, y_value, n_epochs=n_epochs, lr=lr, momentum=momentum, verbose=verbose)

    def save_models(self, verbose=True):
        """Save the policy and value function to dir_path"""
        if verbose:
            print(f"\n>>> Saving models in {self.dir_path}")
        torch.save(self.policy, f"{self.dir_path}\\policy.pkl")
        torch.save(self.value_function, f"{self.dir_path}\\value_function.pkl")
