from blokus.envs.shapes.shapes import get_all_shapes
from blokus.envs.blokus_env import BlokusEnv


class BlokusSimpleEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 2
    BOARD_SIZE = 7
    STATES_FILE = "states_simple.json"
    all_shapes = [shape for shape in get_all_shapes() if shape.size < 5]
