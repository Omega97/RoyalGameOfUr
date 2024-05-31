""" # todo read this, then run the program!
Ciao Luis!

There is a lot of code in this repo, so let's get started! The good news is
that you don't have to implement everything from scratch. You just need to
implement the LuisAgent class by subclassing my NNAgent. ;)
    The main method you need to implement is the get_action method. It takes
as input the state information, and returns both the move that the bot makes
(int in this case), and also the evaluation of the state from the point of
view of both players (np.array).
    If you run this program, it should basically already work. The games and
models should be saved in the luis_ur_games and luis_ur_models respectively.
    Please work on this file only. If there are other things that you need to
change about the NNAgent, please do so in your LuisAgent. If there are things
you need to modify in the rest of the repo, please contact the closest Omar
operator in your area. XD

Final remarks: The current implementation has a few weaknesses.
1) The policy augmented with the rollouts isn't significantly  stronger then
the row policy, and this makes the training  slow (to say the least!) :(
2) Longer rollouts result in a better initial learning (for obvious reasons),
but become quickly a computational burden, and result in more random
evaluations later in the learning process. :(
"""
import os
import torch
import torch.nn as nn
from nn_agent import NNAgent
from royal_game_of_ur import RoyalGameOfUr
from training import Training


def initialize_weights(model, weight_range=0.1):
    """ Initialize weights within the given range
    # todo: you can modify this function if you want, but you don't really have to.
    """
    assert weight_range > 0, "Weight range must be positive"
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -weight_range, weight_range)
            torch.nn.init.uniform_(layer.bias, -weight_range, weight_range)


class LuisAgent(NNAgent):
    """Your very own implementation of the Ur bot"""
    def __init__(self, game_instance, dir_path, n_rollouts, rollout_depth,
                 reward_half_life, hidden_units=100, greedy=False, verbose=False):
        super().__init__(game_instance=game_instance,
                         dir_path=dir_path,
                         n_rollouts=n_rollouts,
                         rollout_depth=rollout_depth,
                         reward_half_life=reward_half_life,
                         hidden_units=hidden_units,
                         greedy=greedy,
                         verbose=verbose)

    def __repr__(self):
        return (f"LuisAgent(n_rollouts={self.n_rollouts}, "
                f"rollout_depth={self.rollout_depth}, "
                f"greedy={self.greedy})")

    def reset_policy(self, input_size, hidden_units, output_size):
        """ MLP Policy
        # todo: this is how I initialize the policy network. Feel free to play with it.
        """
        self.policy = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Sigmoid(),
                nn.Linear(hidden_units, output_size),
                nn.Softmax(dim=1)
            ))

    def reset_value_function(self, input_size, hidden_units, output_size, weight_range=0.01):
        """ MLP value function
        # todo: this is how I initialize the value function. Feel free to play with it.
        """
        self.value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Sigmoid(),
                nn.Linear(hidden_units, output_size),
                nn.Sigmoid()
            ))
        initialize_weights(self.value_function, weight_range)

    def _reset_search(self, state_info):
        """Reset the search for a new state
        # todo: if you do something fancy with the tree search, here is where you can initialize it.
        """
        super()._reset_search(state_info)

    def get_action(self, state_info: dict, verbose=False, bar_length=30, half_life=20) -> dict:
        """Get the best action for the current state
        state_info: dictionary of {"current_player": self.current_player,   # player id, 0 or 1
                                   "n_steps": self.n_steps,                 # roll, int >= 0
                                   "round": self.round,                     # move counter, int >= 0
                                   "board": self.board,                     # np.array of shape (2, 15)
                                   "legal_moves": self.get_legal_moves()}   # binary np.array of len 16
        returns: dictionary of {"action": action,                           # int
                                "eval": evaluation}                         # your evaluation (expected reward, result of search) of the current state, np.array of len 2
        # todo: this is the heart and mind of the agent. Here is where you should implement the bot.
        """
        return super().get_action(state_info, verbose=verbose,
                                  bar_length=bar_length, half_life=half_life)


def main():
    root_dir = os.getcwd()

    agent = LuisAgent(game_instance=RoyalGameOfUr(),
                      dir_path=os.path.join(root_dir, 'luis_ur_models'),
                      n_rollouts=30,
                      rollout_depth=50,
                      reward_half_life=20
                      )

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'luis_ur_games')
                        )

    training.run(n_cycles=99,
                 n_games_per_cycle=10,
                 n_epochs_policy=300,
                 n_epochs_value=300,
                 halflife=1,
                 lr=0.1,
                 verbose=False)


if __name__ == '__main__':
    main()
