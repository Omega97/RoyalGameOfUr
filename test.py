import os
from royal_game_of_ur import RoyalGameOfUr
from example_states_test import STATES
from nn_agent import NNAgent
from evaluation import evaluation_match


def test_states(root_dir=os.getcwd()):

    agent = NNAgent(game_instance=RoyalGameOfUr(),
                    dir_path=os.path.join(root_dir, 'ur_models'),
                    n_rollouts=200,
                    rollout_depth=3,
                    verbose=True)

    states = [STATES[3]]

    for i, state in enumerate(states):
        print(f'\n\nState: {i}\n')
        game = RoyalGameOfUr().set_state(state)
        print(game)

        agent.get_action(game.get_state_info(), verbose=True)


def test_parameters(root_dir=os.getcwd()):

    state = STATES[2]
    game = RoyalGameOfUr().set_state(state)
    player = game.get_state_info()["current_player"]
    print(game)

    for i in range(10):
        n = 2**(i+4)
        agent = NNAgent(game_instance=RoyalGameOfUr(),
                        dir_path=os.path.join(root_dir, 'ur_models'),
                        n_rollouts=n,
                        rollout_depth=3,
                        verbose=False)
        action = agent.get_action(game.get_state_info(), verbose=True)
        print(f'{n:>2})  '
              f'action = {action["action"]}   '
              f'eval = {action["eval"][player]:5.1%}\n')


def compare_agents(root_dir=os.getcwd()):
    """Play a game between two agents and print the result."""
    agents = list()

    agents.append(NNAgent(game_instance=RoyalGameOfUr(),
                          dir_path=os.path.join(root_dir, 'ur_models'),
                          n_rollouts=100,
                          rollout_depth=3))

    agents.append(NNAgent(game_instance=RoyalGameOfUr(),
                          dir_path=os.path.join(root_dir, 'ur_models'),
                          n_rollouts=30,
                          rollout_depth=3))

    evaluation_match(agents[0], agents[1], n_games=200, verbose=True)
    evaluation_match(agents[1], agents[0], n_games=200, verbose=True)


if __name__ == '__main__':
    # test_states()
    # test_parameters()
    compare_agents()
