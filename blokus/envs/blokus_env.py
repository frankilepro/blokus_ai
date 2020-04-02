# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import blokus.envs.shapes.shapes as shapes
from blokus.envs.players.player import Player
from blokus.envs.players.random_player import Random_Player
from blokus.envs.game.board import Board
from blokus.envs.game.blokus_game import BlokusGame
import matplotlib.pyplot as plt


class BlokusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        All_Shapes = [shapes.I1(), shapes.I2(), shapes.I3(), shapes.I4(), shapes.I5(),
                      shapes.V3(), shapes.L4(), shapes.Z4(), shapes.O4(), shapes.L5(),
                      shapes.T5(), shapes.V5(), shapes.N(), shapes.Z5(), shapes.T4(),
                      shapes.P(), shapes.W(), shapes.U(), shapes.F(), shapes.X(), shapes.Y()]

        first = Player("A", "Computer_A", Random_Player)
        second = Player("B", "Computer_B", Random_Player)
        third = Player("C", "Computer_C", Random_Player)
        fourth = Player("D", "Computer_D", Random_Player)
        standard_size = Board(14, 14, "_")
        ordering = [first, second, third, fourth]
        random.shuffle(ordering)
        self.blokus_game = BlokusGame(ordering, standard_size, All_Shapes)

    def step(self, action):
        self.blokus_game.play()
        done = self.blokus_game.winner() != "None"
        return {}, 0, done, {}

    def reset(self):
        return {}  # TODO

    def render(self, mode='human'):
        self.blokus_game.board.print_board(num=self.blokus_game.rounds)

    def close(self):
        plt.close('all')
