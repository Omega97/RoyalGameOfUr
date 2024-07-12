import os
from game.training import Training
from agents.nn_agent import NNAgent
from game.royal_game_of_ur import RoyalGameOfUr
# todo save player on duty
# todo save state to record when setting state
# todo rollouts in parallel
# todo if there is data, start with training
# todo group the generated games and models by iteration
# todo run torch on GPU
# todo merge policy with value function
# todo rollouts return weighted average of each state of the trajectory


def main():
    root_dir = os.getcwd()

    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    dir_path=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=100,
                    rollout_depth=5,
                    )

    training = Training(agent_instance=agent,
                        game_instance=RoyalGameOfUr(),
                        games_dir=os.path.join(root_dir, 'ur_games')
                        )

    training.run(n_cycles=99,
                 n_games_per_cycle=20,
                 n_epochs_policy=500,
                 n_epochs_value=500,
                 halflife=1,
                 lr=0.01,
                 verbose=False)


if __name__ == '__main__':
    main()
