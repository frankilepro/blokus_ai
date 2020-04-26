from blokus_gym.envs.blokus_env import BlokusEnv
from blokus_gym.envs.shapes.shapes import get_all_shapes
from blokus_gym.envs.players.greedy_player import GreedyPlayer
from blokus_gym.envs.players.minimax_player import MinimaxPlayer


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


class BlokusSimpleGreedyEnv(BlokusSimpleEnv):
    bot_type = MinimaxPlayer
