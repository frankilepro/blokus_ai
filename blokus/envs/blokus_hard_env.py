# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import gym
import os
from gym import error, spaces, utils
from gym.utils import seeding
import blokus.envs.shapes.shapes as shapes
from blokus.envs.players.player import Player
from blokus.envs.players.random_player import Random_Player
from blokus.envs.game.board import Board
from blokus.envs.game.blokus_game import BlokusGame
import matplotlib.pyplot as plt
import json


class BlokusHardEnv(gym.Env):
    STATES_FILE = "states.json"
    metadata = {'render.modes': ['human']}

    def __init__(self):
        All_Shapes = [shapes.I1(), shapes.I2(), shapes.I3(), shapes.I4(), shapes.I5(),
                      shapes.V3(), shapes.L4(), shapes.Z4(), shapes.O4(), shapes.L5(),
                      shapes.T5(), shapes.V5(), shapes.N(), shapes.Z5(), shapes.T4(),
                      shapes.P(), shapes.W(), shapes.U(), shapes.F(), shapes.X(), shapes.Y()]

        self.ai = Player("A", "ai_strategy_name", Random_Player)
        second = Player("B", "Computer_B", Random_Player)
        third = Player("C", "Computer_C", Random_Player)
        fourth = Player("D", "Computer_D", Random_Player)
        standard_size = Board(21, 21, "_")
        ordering = [self.ai, second, third, fourth]
        ordering = [self.ai, second]
        random.shuffle(ordering)
        self.blokus_game = BlokusGame(ordering, standard_size, All_Shapes)

        self.observation_space = spaces.Box(0, 4, (21, 21), dtype=int)   # Nothing, us or them on every tile
        self.set_all_possible_moves()
        self.action_space = spaces.Discrete(len(self.all_possible_moves))

    def step(self, action):
        self.ai.strategy = lambda player, game: action
        self.blokus_game.play()

        while self.blokus_game.next_player() != self.ai:
            self.blokus_game.play()

        done = self.blokus_game.winner() != "None"
        return self.blokus_game.board, 0, done, {}

    def reset(self):
        return {}  # TODO

    def render(self, mode='human'):
        self.blokus_game.board.print_board(num=self.blokus_game.rounds)

    def close(self):
        plt.close('all')

    def ai_possible_moves(self):
        return self.ai.possible_moves([p for p in self.ai.pieces], self.blokus_game)

    def set_all_possible_moves(self):
        if os.path.exists(self.STATES_FILE):
            with open(self.STATES_FILE) as json_file:
                self.all_possible_moves = json.load(json_file)
        else:
            all_possible_moves = self.ai.possible_moves(
                [p for p in self.ai.pieces], self.blokus_game, no_restriction=True)
            data = {str(move): idx for idx, move in enumerate(all_possible_moves)}
            with open(self.STATES_FILE, "w") as json_file:
                json.dump(data, json_file, indent=4)
            self.blokus_game.start()
