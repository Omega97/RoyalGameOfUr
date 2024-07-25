import os
import torch
from torch import nn
from game.training import Training
from agents.nn_value_agent import DeepNNValueAgent
from game.royal_game_of_ur import RoyalGameOfUr
from src.utils import initialize_weights, WeightedMSELoss
from tests.demo_game import demo_game_nn_value_agent


def main(n_cycles=10,
         weight_decay=1e-6,
         n_evaluation_games=0,              # set to 0 to skip evaluation
         criterion_class=WeightedMSELoss,   # nn.modules.loss.MSELoss, WeightedMSELoss
         optimizer_class=torch.optim.Adam,  # Adagrad, Adam
         n_games_per_cycle=250,
         n_epoch=300,
         model_hidden_units=100,
         model_weight_range=0.2,
         ):
    """
    This method creates a model from scratch (or continues the training
    of an already existing model). The training process is organized
    in three steps, during which the learning rate is decreased
    """
    root_dir = os.getcwd()

    # Define the architecture of the value function
    def value_function_builder(input_size, output_size,
                               hidden_units=model_hidden_units*2,
                               weight_range=model_weight_range):
        """
        Build a neural network with input_size input units,
        hidden_units hidden units, and output_size output units
        """
        value_function = (
            nn.Sequential(
                nn.Linear(input_size, hidden_units),
                nn.Softplus(),
                nn.Dropout(p=0.5),  # Dropout layer with 50% probability
                nn.Linear(hidden_units, hidden_units),
                nn.Softplus(),
                nn.Dropout(p=0.5),  # Dropout layer with 50% probability
                nn.Linear(hidden_units, hidden_units//5),
                nn.Softplus(),
                nn.Dropout(p=0.5),  # Dropout layer with 50% probability
                nn.Linear(hidden_units//5, output_size),
                nn.Sigmoid()
            )
        )
        initialize_weights(value_function, weight_range=weight_range)
        return value_function

    # Create the agent
    agent = DeepNNValueAgent(game_instance=RoyalGameOfUr(), depth=3,
                             models_dir_path=os.path.join(root_dir, 'ur_models'),
                             value_function_builder=value_function_builder)

    # Initialize the training process
    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games'))

    # Run the first phase training process
    # training.run(n_cycles=20,
    #              n_games_per_cycle=n_games_per_cycle,
    #              weight_decay=weight_decay,
    #              n_evaluation_games=n_evaluation_games,
    #              criterion_class=criterion_class,
    #              optimizer_class=optimizer_class,
    #              n_epoch=n_epoch,
    #              halflife=20,       # 20
    #              lr=1e-3,           # 1e-3
    #              )

    # Run the second phase training process
    # training.run(n_cycles=20,
    #              n_games_per_cycle=n_games_per_cycle,
    #              weight_decay=weight_decay,
    #              n_evaluation_games=n_evaluation_games,
    #              criterion_class=criterion_class,
    #              optimizer_class=optimizer_class,
    #              n_epoch=n_epoch,
    #              halflife=0,        # 0
    #              lr=1e-4,           # 1e-4
    #              )

    # Refinement
    training.run(n_cycles=1000,     # 1000
                 n_games_per_cycle=0,
                 weight_decay=weight_decay,
                 n_evaluation_games=n_evaluation_games,
                 criterion_class=criterion_class,
                 optimizer_class=optimizer_class,
                 n_epoch=1000,    # 300
                 halflife=0,     # 0
                 lr=1e-4,        # 1e-3
                 do_delete_games=False
                 )

    # Evaluation
    # training.evaluate_agent(300)

    # demo_game_nn_value_agent(dir_path='ur_models')


if __name__ == '__main__':
    main()
