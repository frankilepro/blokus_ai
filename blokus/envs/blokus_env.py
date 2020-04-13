# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import gym
import os
from gym import error, spaces, utils
from gym.utils import seeding
import blokus.envs.shapes.shapes as shapes
from blokus.envs.shapes.shape import Shape
from blokus.envs.players.player import Player
from blokus.envs.players.random_player import Random_Player
from blokus.envs.game.board import Board
from blokus.envs.game.blokus_game import BlokusGame
import matplotlib.pyplot as plt
import json


class BlokusEnv(gym.Env):
    STATES_FILE = "states.json"
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.init_game()

    def init_game(self):
        All_Shapes = [shapes.I1(), shapes.I2(), shapes.I3(), shapes.I4(), shapes.I5(),
                      shapes.V3(), shapes.L4(), shapes.Z4(), shapes.O4(), shapes.L5(),
                      shapes.T5(), shapes.V5(), shapes.N(), shapes.Z5(), shapes.T4(),
                      shapes.P(), shapes.W(), shapes.U(), shapes.F(), shapes.X(), shapes.Y()]

        self.ai = Player("A", "ai", Random_Player)
        second = Player("B", "Computer_B", Random_Player)
        standard_size = Board(14, 14, "_")
        ordering = [self.ai, second]
        random.shuffle(ordering)
        self.blokus_game = BlokusGame(ordering, standard_size, All_Shapes)

        self.observation_space = spaces.Box(0, 2, (14, 14), dtype=int)  # Nothing, us or them on every tile
        self.set_all_possible_moves()
        self.action_space = spaces.Discrete(len(self.all_possible_ids_to_moves))

    def step(self, action_id):
        self.ai.strategy = lambda player, game: None if action_id is None else self.all_possible_ids_to_moves[action_id]
        self.blokus_game.play()

        done, reward = self.get_done_reward()
        while not done and self.blokus_game.next_player() != self.ai:
            self.blokus_game.play()
            done, reward = self.get_done_reward()

        done, reward = self.get_done_reward()
        return self.blokus_game.board.tensor, reward, done, {}

    def get_done_reward(self):
        winner = self.blokus_game.winner()
        done = winner != "None"
        if done:
            if winner == "ai":
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        return done, reward

    def reset(self):
        self.init_game()
        return self.blokus_game.board.tensor

    def render(self, mode='human'):
        self.blokus_game.board.print_board(num=self.blokus_game.rounds, mode=mode)

    def close(self):
        plt.close('all')

    def ai_possible_ids(self):
        possible_moves = self.ai.possible_moves([p for p in self.ai.pieces], self.blokus_game)
        possible_ids = [self.all_possible_moves_to_ids[move] for move in possible_moves]
        return possible_ids

    def set_all_possible_moves(self):
        if os.path.exists(self.STATES_FILE):
            with open(self.STATES_FILE) as json_file:
                self.all_possible_ids_to_moves = [Shape.from_json(move) for move in json.load(json_file)]
        else:
            self.all_possible_ids_to_moves = self.ai.possible_moves(
                [p for p in self.ai.pieces], self.blokus_game, no_restriction=True)
            data = [move.to_json() for move in self.all_possible_ids_to_moves]
            with open(self.STATES_FILE, "w") as json_file:
                json.dump(data, json_file)
            self.blokus_game.start()

        self.all_possible_moves_to_ids = {}
        for idx, move in enumerate(self.all_possible_ids_to_moves):
            self.all_possible_moves_to_ids[move] = idx
