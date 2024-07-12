import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from game.evaluation import evaluation_match
from game.agent import Agent
from agents.random_agent import RandomAgent
from agents.training_data import state_to_features
from src.utils import bar, bcolors


def initialize_weights(model, weight_range=0.1):
    """ Initialize weights within the given range """
    assert weight_range > 0, "Weight range must be positive"
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)


class NNValueAgent(Agent):

    def __init__(self,
                 game_instance,
                 dir_path,
                 n_rollouts=100,
                 rollout_depth=5,
                 hidden_units=100,
                 reward_half_life=20,
                 greedy=False,
                 verbose=False):
        self.game = game_instance
        self.dir_path = dir_path
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.greedy = greedy
        self.gamma = np.log(2) / reward_half_life

        self.game_input_size = game_instance.INPUT_SIZE
        self.value_function = None
        self.legal_indices = None
        self.visits = None
        self.scores = None
        self.states = None
        self.reset(hidden_units=hidden_units, verbose=verbose)

    def __repr__(self):
        return (f"NewNNValueAgent(n_rollouts={self.n_rollouts}, "
                f"rollout_depth={self.rollout_depth}, "
                f"greedy={self.greedy})")

    def reset_value_function(self, input_size, hidden_units, output_size, weight_range=0.01):
        """ MLP value function """
        print('Resetting value function')
        self.value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, output_size),
                nn.Sigmoid()
            ))
        initialize_weights(self.value_function, weight_range)

    def _get_move_values(self, legal_moves, legal_move_indices, state_info, player_id):
        """Get the value of each move in legal_moves"""
        values = np.zeros(len(legal_moves))

        # iterate over all legal moves
        for i in range(len(legal_move_indices)):
            state = self.game.deepcopy()
            state = state.set_state(state_info)
            move = legal_move_indices[i]

            # get new state after move
            state.move(move)

            # evaluate state
            new_value = 0.
            probabilities = np.array([1, 4, 6, 4, 1]) / 16
            # iterate over all rolls
            for j in range(len(probabilities)):
                state_2 = state.deepcopy()
                state_2.n_steps = j
                state_2_info = state_2.get_state_info()
                features = state_to_features(state_2_info)
                x = np.array([features])
                x = torch.tensor(x, dtype=torch.float)
                if state_2.is_game_over():
                    new_value += state_2.get_reward()[player_id] * probabilities[j]
                else:
                    y = self.value_function(x).detach().numpy()
                    y = y[0]
                    new_value += y[player_id] * probabilities[j]

            values[i] = new_value

        return values

    def reset(self, hidden_units=100, verbose=False):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        try:
            # print(f'\n>>> Trying to load models from {self.get_value_function_path()}')
            self.value_function = torch.load(self.get_value_function_path())
            # print('>>> Models loaded successfully!')
        except FileNotFoundError:
            print('\n>>> Loading failed. Creating new models...')
            self.reset_value_function(input_size=self.game_input_size,
                                      hidden_units=hidden_units,
                                      output_size=2)
            torch.save(self.value_function, self.get_value_function_path())
            print('>>> Value function created')

    def _reset_search(self, state_info):
        """Reset the search for a new state"""
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

    def get_value_function_path(self):
        return os.path.join(self.dir_path, "value_function.pkl")

    def evaluate_state(self, state_info: dict):
        """
        Evaluate a state using the value function
        """
        features = state_to_features(state_info)
        x = np.array([features])
        x = torch.tensor(x, dtype=torch.float)
        y = self.value_function(x).detach().numpy()
        return y[0]

    def _get_random_rollout_depth(self):
        """
        Get a random number between 1 and self.rollout_depth, or
        None if self.rollout_depth is None.
        """
        return np.random.randint(1, self.rollout_depth + 1)

    def _get_output(self, state_info: dict):
        """After all the process to get the action and state evaluation
        is completed, return the action and evaluation """
        values = self.scores / self.visits
        best_index = int(np.argmax(values))
        evaluation = values[best_index]
        evaluation = [evaluation, 1 - evaluation] if state_info["current_player"] == 0 else [1 - evaluation, evaluation]
        action = self.legal_indices[best_index]
        return {"action": action, "eval": evaluation}

    def _print_search_results(self, state_info, ev, t, bar_length):
        color = 91 if state_info["current_player"] == 1 else 94
        best_value_index = int(np.argmax(ev))

        print('\nSearch EV:')
        for i in range(self.n_moves):
            p = ev[i]
            s = f'{self.legal_indices[i]:3})  {p:6.2%}  '
            s += bar(p, color, length=bar_length)
            if i == best_value_index:
                s += ' <<'
            print(s)

        print(f"\nTime: {t:.2f} s")

    def get_action(self, state_info: dict, verbose=False, bar_length=30, half_life=10) -> dict:
        """
        Evaluate all the possible moves and return the one
        with the highest expected value.
        """
        assert self.value_function is not None
        player_id = state_info["current_player"]
        legal_moves = state_info["legal_moves"]
        legal_move_indices = np.arange(len(legal_moves))[legal_moves > 0]
        values = self._get_move_values(legal_moves, legal_move_indices, state_info, player_id)

        # get best move
        best_move = legal_move_indices[np.argmax(values)]
        best_eval = max(values)
        eval_ = [best_eval, 1 - best_eval]
        if player_id == 1:
            eval_ = list(reversed(eval_))
        eval_ = np.array(eval_)

        if verbose:
            print(f'action {best_move}')
            color = bcolors.RED if state_info["current_player"] == 1 else bcolors.BLUE
            print(f'{bar(p=best_eval, color=color, length=bar_length)} {best_eval:.3f}')

        return {"action": best_move, "eval": eval_}

    def get_value_function(self):
        return self.value_function

    def _get_value_function_rmse(self, x, y_true):
        """ example tensors -> rmse"""
        y = y_true.detach().numpy()
        y_pred = self.value_function(x)
        rmse = np.sqrt(np.mean((y_pred.detach().numpy() - y) ** 2))
        return rmse

    def train_value_function(self, x, y,
                             n_epochs=1000, lr=0.1,
                             momentum=0.9, batch_size=100,
                             print_period=1, verbose=True):
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
        rmse_before = self._get_value_function_rmse(x, y)
        if verbose:
            print(f'     rmse: {rmse_before:.5f}')

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
                print(f'\rEpoch {epoch + 1:5}, Loss: {np.average(losses):.6f}', end=' ')

        # rmse
        rmse_after = self._get_value_function_rmse(x, y)
        if verbose:
            print(f'\n     rmse: {rmse_after:.5f}')

        # save value function
        if rmse_after < rmse_before:
            print('\033[92m' + f'Saving value function' + '\033[0m')
            torch.save(self.value_function, self.get_value_function_path())
            print(self.get_value_function_path())
        else:
            print('\033[91m' + 'RMSE did not improve. Value function not saved' + '\033[0m')

        return rmse_after

    def train_agent(self, x, y_policy, y_value, n_epochs=500,
                    lr=0.1, batch_size=200,
                    momentum=0.9, verbose=True):
        """Train the value function using the training data"""
        performance = self.train_value_function(x, y_value, n_epochs=n_epochs,
                                                lr=lr,  batch_size=batch_size,
                                                momentum=momentum, verbose=verbose)
        return performance

    def evaluate(self, n_games=500, show_game=False):
        """ Evaluate the components of the agent (value function) """
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
