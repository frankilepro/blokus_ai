from blokus.envs.blokus_env import BlokusEnv


class BlokusDuoEnv(BlokusEnv):
    NUMBER_OF_PLAYERS = 2
    BOARD_SIZE = 14
    STATES_FILE = "states_duo.json"
