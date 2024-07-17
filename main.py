import os
from torch import nn
from game.training import Training
from agents.nn_value_agent import NNValueAgent
from game.royal_game_of_ur import RoyalGameOfUr
from src.utils import initialize_weights


def main():
    root_dir = os.getcwd()

    # Define the architecture of the value function
    def value_function_builder(input_size, output_size,
                               hidden_units=100, weight_range=0.1):
        """
        Build a neural network with input_size input units,
        hidden_units hidden units, and output_size output units
        """
        value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Softplus(),
                nn.Linear(hidden_units, hidden_units),
                nn.Softplus(),
                nn.Linear(hidden_units, output_size),
                nn.Sigmoid()
            ))
        initialize_weights(value_function, weight_range=weight_range)
        return value_function

    # Create the agent
    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path=os.path.join(root_dir, 'ur_models'),
                         value_function_builder=value_function_builder)

    # Initialize the training process
    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    # Run the training process
    training.run(n_cycles=100,
                 n_games_per_cycle=200,
                 halflife=5)


if __name__ == '__main__':
    main()
