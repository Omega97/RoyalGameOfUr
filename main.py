import os
from training import Training
from nn_agent import NNAgent
from royal_game_of_ur import RoyalGameOfUr
# todo save state to record when setting state
# todo rollouts in parallel
# todo if there is data, start with training


def main(root_dir='C:\\Users\\monfalcone\\PycharmProjects\\ReinforcementLearning'):

    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    models_dir=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=100,
                    rollout_depth=5)

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games')
                        )

    training.run(n_cycles=1, n_games_per_cycle=10, halflife=10, verbose=False)


if __name__ == '__main__':
    main()