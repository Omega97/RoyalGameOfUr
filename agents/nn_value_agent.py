import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from game.evaluation import evaluation_match
from game.agent import Agent
from agents.random_agent import RandomAgent
from game.training_data import state_to_features
from src.utils import bar, bcolors, cprint


def initialize_weights(model, weight_range=0.1):
    """ Initialize weights within the given range """
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
                 hidden_units=100,
                 reward_half_life=20,
                 verbose=False):

        self.game = game_instance
        self.dir_path = models_dir_path
        self.depth = depth
        self.gamma = np.log(2) / reward_half_life
        self.game_input_size = game_instance.INPUT_SIZE

        self.value_function = None
        self.legal_indices = None
        self.visits = None
        self.scores = None
        self.states = None

        self.reset(hidden_units=hidden_units, verbose=verbose)

    def __repr__(self):
        return f"NewNNValueAgent(depth={self.depth})"

    def reset_value_function(self, input_size, hidden_units, output_size, weight_range=0.01):
        """ MLP value function """
        cprint('Resetting value function', bcolors.WARNING)
        self.value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Sigmoid(),
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

            # iterate over all rolls
            for j in range(len(NNValueAgent.PROBA)):
                state_2 = state.deepcopy()
                state_2.n_steps = j
                state_2_info = state_2.get_state_info()
                features = state_to_features(state_2_info)
                x = np.array([features])
                x = torch.tensor(x, dtype=torch.float)
                if state_2.is_game_over():
                    new_value += state_2.get_reward()[player_id] * NNValueAgent.PROBA[j]
                else:
                    y = self.value_function(x).detach().numpy()
                    y = y[0]
                    new_value += y[player_id] * NNValueAgent.PROBA[j]

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
            cprint('\n>>> Loading failed. Creating new models...', bcolors.WARNING)
            self.reset_value_function(input_size=self.game_input_size,
                                      hidden_units=hidden_units,
                                      output_size=2)
            torch.save(self.value_function, self.get_value_function_path())
            cprint('>>> Value function created', bcolors.OKGREEN)

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
        """Print the results of the search"""
        color = bcolors.RED if state_info["current_player"] == 1 else bcolors.BLUE
        best_value_index = int(np.argmax(ev))

        cprint('\nSearch EV:')
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

    def train_value_function(self, x, y, lr=0.1, momentum=0.9, verbose=True):
        """Train the value function using the training data"""
        if self.value_function is None:
            raise ValueError("Value function is not set")
        elif verbose:
            cprint("\n>>> Training value function", bcolors.WARNING)

        # Instantiate loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.value_function.parameters(), lr=lr, momentum=momentum)

        # update
        optimizer.zero_grad()
        output = self.value_function(x.float())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # save value function
        print(f'Saving value function to {self.get_value_function_path()}')
        torch.save(self.value_function, self.get_value_function_path())
        cprint(f'Function saved', bcolors.OKGREEN)

    def train_agent(self, x, y_policy, y_value, lr=0.1, verbose=True):
        """Train the value function using the training data"""
        return self.train_value_function(x, y_value, lr=lr, verbose=verbose)

    def evaluate(self, n_games=500, show_game=False):
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
