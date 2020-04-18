# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# Inspired from https://github.com/mknapper1/Machine-Learning-Blokus
import json
import matplotlib.pyplot as plt
from blokus.envs.game.game import InvalidMoveByAi
from blokus.envs.game.blokus_game import BlokusGame
from blokus.envs.game.board import Board
from blokus.envs.players.random_player import Random_Player
from blokus.envs.players.player import Player
from blokus.envs.shapes.shape import Shape
import blokus.envs.shapes.shapes as shapes
from gym.utils import seeding
from gym import error, spaces, utils
import os
import gym
import random


class BlokusEnv(gym.Env):
    STATES_FILE = "states.json"
    metadata = {'render.modes': ['human']}
    rewards = {'default': 0, 'won': 1, 'invalid': -1, 'lost': -2}

    def __init__(self):
        self.init_game()

    def init_game(self):
        self.all_shapes = [shapes.I1(), shapes.I2(), shapes.I3(), shapes.I4(), shapes.I5(),
                           shapes.V3(), shapes.L4(), shapes.Z4(), shapes.O4(), shapes.L5(),
                           shapes.T5(), shapes.V5(), shapes.N(), shapes.Z5(), shapes.T4(),
                           shapes.P(), shapes.W(), shapes.U(), shapes.F(), shapes.X(), shapes.Y()]

        standard_size = Board(14, 14, "_")
        self.blokus_game = BlokusGame(standard_size, self.all_shapes)

        self.observation_space = spaces.Box(0, 2, (14, 14), dtype=int)  # Nothing, us or them on every tile
        self.__set_all_possible_moves()
        self.action_space = spaces.Discrete(len(self.all_possible_indexes_to_moves))
        self.action_space.sample = self.ai_sample_possible_index

        self.ai = Player("A", "ai", Random_Player, self.all_possible_indexes_to_moves, self.blokus_game)
        second = Player("B", "Computer_B", Random_Player, self.all_possible_indexes_to_moves, self.blokus_game)
        ordering = [self.ai, second]
        random.shuffle(ordering)
        for player in ordering:
            self.blokus_game.add_player(player)

        while self.blokus_game.next_player() != self.ai:
            self.__next_player_play()  # Let bots start

    def step(self, action_id):
        self.__set_ai_strategy(action_id)

        done, reward = self.__next_player_play()  # Let ai play
        while not done and self.blokus_game.next_player() != self.ai:
            done, reward = self.__next_player_play()  # Let bots play

        if not self.ai.remains_move and not done:
            while not done:
                done, reward = self.__next_player_play()  # If ai has no move left, let the game finish

        return self.blokus_game.board.tensor, reward, done, {'valid_actions': self.ai_possible_mask()}

    def __set_ai_strategy(self, action_id):
        self.ai.strategy = lambda player:\
            None if not self.ai.remains_move else self.all_possible_indexes_to_moves[action_id]

    def __next_player_play(self):
        try:
            self.blokus_game.play()
            return self.__get_done_reward()
        except InvalidMoveByAi:
            self.__set_ai_strategy(self.ai_sample_possible_index())
            self.blokus_game.play()
            done, reward = self.__get_done_reward()
            return done, min(self.rewards['invalid'], reward)

    def __get_done_reward(self):
        winner = self.blokus_game.winner()
        done = winner is not None
        if done:
            if winner == "ai":
                reward = self.rewards['won']
            else:
                reward = self.rewards['lost']
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

    def ai_sample_possible_index(self):
        actions = self.ai_possible_indexes()
        return random.choice(actions)

    def ai_possible_indexes(self):
        return self.ai.possible_move_indexes()

    def ai_possible_mask(self):
        indexes = self.ai.possible_move_indexes()
        mask = [False] * len(self.all_possible_indexes_to_moves)
        for index in indexes:
            mask[index] = True
        return mask

    def __set_all_possible_moves(self):
        if os.path.exists(self.STATES_FILE):
            with open(self.STATES_FILE) as json_file:
                self.all_possible_indexes_to_moves = [Shape.from_json(move) for move in json.load(json_file)]
        else:
            print("Building all possible state, this may take some time")
            dummy = Player("", "", None, self.all_shapes, self.blokus_game)
            self.all_possible_indexes_to_moves = dummy.possible_moves([p for p in self.all_shapes], no_restriction=True)
            data = [move.to_json(idx) for idx, move in enumerate(self.all_possible_indexes_to_moves)]
            with open(self.STATES_FILE, "w") as json_file:
                json.dump(data, json_file)

        self.all_possible_moves_to_indexes = {}
        for move in self.all_possible_indexes_to_moves:
            self.all_possible_moves_to_indexes[move] = move.idx
