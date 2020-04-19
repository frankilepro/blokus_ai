# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# Inspired from https://github.com/mknapper1/Machine-Learning-Blokus
import json
import multiprocessing as mp
import itertools
from functools import partial
import matplotlib.pyplot as plt
from blokus.envs.game.game import InvalidMoveByAi
from blokus.envs.game.blokus_game import BlokusGame
from blokus.envs.game.board import Board
from blokus.envs.players.random_player import Random_Player
from blokus.envs.players.player import Player
from blokus.envs.shapes.shape import Shape
from blokus.envs.shapes.shapes import get_all_shapes
from gym.utils import seeding
from gym import error, spaces, utils
import os
import cython
import gym
import random


def possible_moves_func(dummy, board_size, pieces):
    # This needs to be there because it can't be pickled
    return dummy.possible_moves(pieces, no_restriction=True, board_size=board_size)


class BlokusEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    rewards = {'won': 2, 'tie-won': 0, 'default': 0, 'invalid': -1, 'lost': -2}
    STATES_FOLDER = "states"

    # Customization available by base classes
    NUMBER_OF_PLAYERS = 4
    BOARD_SIZE = 21
    STATES_FILE = "states.json"
    all_shapes = get_all_shapes()

    def __init__(self):
        assert 2 <= self.NUMBER_OF_PLAYERS <= 4, "Between 2 and 3 players"
        print(f"Is running cython version: {cython.compiled}")
        if not cython.compiled:
            print("You should run 'python setup.py build_ext --inplace' to get a 3x speedup")
        self.all_possible_indexes_to_moves = None
        self.init_game()

    def init_game(self):
        standard_size = Board(self.BOARD_SIZE, self.BOARD_SIZE, "_")
        self.blokus_game = BlokusGame(standard_size, self.all_shapes)

        self.observation_space = spaces.Box(0, self.NUMBER_OF_PLAYERS, (self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        self.__set_all_possible_moves()
        self.action_space = spaces.Discrete(len(self.all_possible_indexes_to_moves))
        self.action_space.sample = self.ai_sample_possible_index

        self.ai = Player(0, "ai", Random_Player, self.all_possible_indexes_to_moves, self.blokus_game)
        bots = [Player(id, f"bot_{id}", Random_Player, self.all_possible_indexes_to_moves, self.blokus_game)
                for id in range(1, self.NUMBER_OF_PLAYERS)]
        ordering = [self.ai] + bots
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
        winners = self.blokus_game.winners()
        done = winners is not None
        if done:
            if "ai" in winners:
                if len(winners) == 1:
                    reward = self.rewards['won']
                else:
                    reward = self.rewards['tie-won']
            else:
                reward = self.rewards['lost']
        else:
            reward = self.rewards['default']
        return done, reward

    def reset(self):
        self.init_game()
        return self.blokus_game.board.tensor

    def render(self, mode='human'):
        self.blokus_game.board.print_board(num=self.blokus_game.rounds, mode=mode)

    def close(self):
        plt.close('all')

    def ai_sample_possible_index(self):
        return self.ai.sample_move_idx()

    def ai_possible_indexes(self):
        return self.ai.possible_move_indexes()

    def ai_possible_mask(self):
        indexes = self.ai.possible_move_indexes()
        mask = [False] * len(self.all_possible_indexes_to_moves)
        for index in indexes:
            mask[index] = True
        return mask

    def __set_all_possible_moves(self):
        if self.all_possible_indexes_to_moves is not None:
            return

        state_file = os.path.join(self.STATES_FOLDER, self.STATES_FILE)
        if os.path.exists(state_file):
            with open(state_file) as json_file:
                self.all_possible_indexes_to_moves = [Shape.from_json(move) for move in json.load(json_file)]
        else:
            print("Building all possible states, this may take some time")
            dummy = Player("", "", None, self.all_shapes, self.blokus_game)

            number_of_cores_to_use = mp.cpu_count() // 2
            with mp.Pool(number_of_cores_to_use) as pool:
                self.all_possible_indexes_to_moves = pool.map(
                    partial(possible_moves_func, dummy, self.BOARD_SIZE), [[p] for p in self.all_shapes])
            self.all_possible_indexes_to_moves = list(itertools.chain.from_iterable(self.all_possible_indexes_to_moves))
            data = [move.to_json(idx) for idx, move in enumerate(self.all_possible_indexes_to_moves)]

            os.makedirs(self.STATES_FOLDER, exist_ok=True)
            with open(state_file, "w") as json_file:
                json.dump(data, json_file)
            print(f"{state_file} has been saved")
