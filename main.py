import os
from game.training import Training
from agents.nn_value_agent import NNValueAgent
from game.royal_game_of_ur import RoyalGameOfUr


def main():
    root_dir = os.getcwd()

    agent = NNValueAgent(game_instance=RoyalGameOfUr(),
                         models_dir_path=os.path.join(root_dir, 'ur_models'),
                         depth=2,
                         )

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games')
                        )

    training.run(n_cycles=1000,
                 n_games_per_cycle=32,
                 halflife=30,
                 lr=.00001,
                 verbose=False)

    # training.evaluate_agent(n_evaluation_games=30)


if __name__ == '__main__':
    main()
