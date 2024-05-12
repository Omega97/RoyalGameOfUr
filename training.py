import os
from copy import deepcopy
from royal_game_of_ur import RoyalGameOfUr
from training_data import create_dataset_from_game_files


class Training:
    """
    An instance of an agent is trained using reinforcement
    learning. The agent uses a policy to make rollouts
    for each move, and a value function to truncate the
    rollouts by evaluating the position after a certain
    number of moves. The agent is trained using the
    following process:

    1) The agents are loaded (policy, value function)
    2) Self-play games are generated by the agent
    playing against itself and stored.
    3) The policy and value function are trained
    using the generated data. The policy must predict
    the action taken by each agent, and the value function
    must predict the expected value of the reward in a
    given state.
    4) Finally, the new policy and value function are saved.

    This process is repeated until the agent converges
    to a good policy and value function.
    """
    def __init__(self, agent_instance,
                 game_instance: RoyalGameOfUr,
                 games_dir):
        """
        Initialize the training process

        agent_instance: instance of the agent class
        game_instance: instance of the game class
        games_dir: directory to store the games played by the agent
        """
        assert hasattr(agent_instance, 'train_agent')
        assert hasattr(agent_instance, 'save_models')  # todo join with training?
        assert hasattr(agent_instance, 'reset')

        self.agent_instance = agent_instance
        self.game = game_instance
        self.games_dir = games_dir

        if not os.path.exists(games_dir):
            os.mkdir(games_dir)

        # data
        self.X = None
        self.y_policy = None
        self.y_value = None

        self.agents = None

    def _load_agents(self):
        """
        Create and set agents
        """
        self.agents = (deepcopy(self.agent_instance),
                       deepcopy(self.agent_instance))
        for agent in self.agents:
            agent.reset()

    def play_game(self, verbose=False):
        """
        Play a game of the Royal Game of Ur. The agent plays
        against itself, and the game is recorded for training.
        The game is then saved to the database directory.
        """
        n = len(os.listdir(self.games_dir)) + 1
        game_copy = self.game.deepcopy()
        game_copy.play(self.agents, verbose=verbose)
        game_copy.save(path=f"{self.games_dir}\\game_{n}.pkl", verbose=verbose)

    def _play_self_play_games(self, n_games, verbose=False):
        """
        Play self-play games and save them to the database
        """
        for i in range(n_games):
            print(f'\nPlaying game {i+1}/{n_games}')
            self.play_game(verbose=verbose)

    def _get_game_files(self, min_n_games, game_ratio=0.5):
        """ Return a list of game files """
        game_files = [f"{self.games_dir}\\{f}" for f in os.listdir(self.games_dir)]
        n_games = int(len(game_files) * game_ratio)
        n_games = max(min_n_games, n_games)
        return game_files[-n_games:]

    def convert_games_to_training_data(self, min_n_games=20, halflife=20):
        """
        Convert the games played by the agent to training data
        and store them in the instance variables X, y_policy, y_value
        """

        # choose files
        game_files = self._get_game_files(min_n_games)
        print(f'\nConverting {len(game_files)} games to training data...')

        # create dataset
        self.X, self.y_policy, self.y_value = create_dataset_from_game_files(game_files=game_files,
                                                                             halflife=halflife)

    def _train_agent(self, n_epochs_policy, n_epochs_value, lr=0.1):
        """
        Train the policy and value function using
        the training data, then save the trained models.
        """
        self.agent_instance.train_agent(x=self.X,
                                        y_policy=self.y_policy,
                                        y_value=self.y_value,
                                        n_epochs_value=n_epochs_value,
                                        n_epochs_policy=n_epochs_policy,
                                        lr=lr)
        self.agent_instance.save_models()

    def run(self, n_cycles, n_games_per_cycle, halflife=20,
            n_epochs_policy=500, n_epochs_value=500, lr=0.1,
            min_n_games=20, verbose=True):
        """ Run the training process """
        for i in range(n_cycles):
            print(f'\n\nTraining cycle {i+1}/{n_cycles}')
            self._load_agents()
            n = n_games_per_cycle  # n = min_n_games if i == 0 else n_games_per_cycle
            self._play_self_play_games(n, verbose=verbose)
            self.convert_games_to_training_data(min_n_games, halflife=halflife)
            self._train_agent(n_epochs_policy, n_epochs_value, lr=lr)
