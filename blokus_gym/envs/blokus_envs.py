from blokus_gym.envs.blokus_env import BlokusEnv
from blokus_gym.envs.shapes.shapes import get_all_shapes
from blokus_gym.envs.players.greedy_player import GreedyPlayer
from blokus_gym.envs.players.minimax_player import MinimaxPlayer
from blokus_gym.envs.players.random_player import RandomPlayer

class BlokusCustomEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 3
    BOARD_SIZE = 10 # This will result in a 10x10 board
    STATES_FILE = "states.json"  # This needs to be set, if not it will take the base class states
    all_shapes = [shape for shape in get_all_shapes()
                  if shape.size == 4]  # This will take only the 4 tiles pieces
    bot_type = RandomPlayer  # Defaults to RandomPlayer if not passed

class BlokusGreedyEnv(BlokusEnv):
    bot_type = GreedyPlayer


class BlokusDuoEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 2
    BOARD_SIZE = 14
    STATES_FILE = "states_duo.json"


class BlokusDuoGreedyEnv(BlokusDuoEnv):
    bot_type = GreedyPlayer


class BlokusSimpleEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 2
    BOARD_SIZE = 7
    STATES_FILE = "states_simple.json"
    all_shapes = [shape for shape in get_all_shapes() if shape.size < 5]


class BlokusSimpleGreedyEnv(BlokusSimpleEnv):
    bot_type = GreedyPlayer


class BlokusSimpleMinimaxEnv(BlokusSimpleEnv):
    bot_type = MinimaxPlayer
