# Class structure follows: https://github.com/openai/gym/blob/master/docs/creating-environments.md
# Inspired from https://github.com/mknapper1/Machine-Learning-Blokus
import json
import multiprocessing as mp
import itertools
from functools import partial
import os
import matplotlib.pyplot as plt
import cython
import gym
from blokus.envs.game.blokus_game import InvalidMoveByAi
from blokus.envs.game.blokus_game import BlokusGame
from blokus.envs.game.board import Board
from blokus.envs.players.ai_player import AiPlayer
from blokus.envs.players.greedy_player import GreedyPlayer
from blokus.envs.players.player import Player
from blokus.envs.shapes.shape import Shape
from blokus.envs.shapes.shapes import get_all_shapes


def possible_moves_func(dummy, board_size, pieces):
    # This needs to be there because it can't be pickled
    return dummy.possible_moves(pieces, no_restriction=True, board_size=board_size)


class BlokusEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    rewards = {'won': 20, 'tie-won': 0, 'default': 0, 'invalid': -100, 'lost': -20}
    STATES_FOLDER = "states"

    # Customization available by base classes
    NUMBER_OF_PLAYERS = 4
    BOARD_SIZE = 21
    STATES_FILE = "states.json"
    all_shapes = get_all_shapes()
    # bot_type = "random"

    def __init__(self):
        assert 2 <= self.NUMBER_OF_PLAYERS <= 4, "Between 2 and 3 players"
        print(f"Is running cython version: {cython.compiled}")
        if not cython.compiled:
            print("You should run 'python setup.py build_ext --inplace' to get a 3x speedup")
        self.all_possible_indexes_to_moves = None
        self.starter_won = 0
        self.last_won = 0
        self.games_played = 0
        self.init_game()

    def init_game(self):
        standard_size = Board(self.BOARD_SIZE)
        self.blokus_game = BlokusGame(standard_size, self.all_shapes)

        self.observation_space = gym.spaces.Box(0, self.NUMBER_OF_PLAYERS,
                                                (self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        self.__set_all_possible_moves()
        self.action_space = gym.spaces.Discrete(len(self.all_possible_indexes_to_moves))
        self.action_space.sample = self.ai_sample_possible_index

        self.ai = AiPlayer(1, "ai", self.all_possible_indexes_to_moves, self.blokus_game)
        bots = [GreedyPlayer(id, f"bot_{id}", self.all_possible_indexes_to_moves,
                             self.blokus_game, deterministic=True)
                for id in range(2, self.NUMBER_OF_PLAYERS + 1)]
        ordering = [self.ai] + bots
        # random.shuffle(ordering)
        for player in ordering:
            self.blokus_game.add_player(player)

        self.starter_player = self.blokus_game.next_player().name
        self.games_played += 1
        while self.blokus_game.next_player() != self.ai:
            self.__next_player_play()  # Let bots start

    def step(self, action_id):
        self.ai.next_move = self.all_possible_indexes_to_moves[action_id]

        done, reward = self.__next_player_play()  # Let ai play
        while not done and self.blokus_game.next_player() != self.ai:
            done, _ = self.__next_player_play()  # Let bots play

        if not done and not self.ai.remains_move:
            self.ai.next_move = None
            while not done:
                done, _ = self.__next_player_play()  # If ai has no move left, let the game finish

        if done:
            done, reward = self.__get_done_reward()

        return self.blokus_game.board.tensor, reward, done, {}
        # return self.blokus_game.board.tensor, reward, done, {'valid_actions': self.ai_possible_mask()}

    def __next_player_play(self):
        try:
            self.blokus_game.play()
            return self.__get_done_reward()
        except InvalidMoveByAi:
            self.ai.next_move = self.ai_sample_possible_index()
            self.blokus_game.play()
            done, reward = self.__get_done_reward()
            return done, min(self.rewards['invalid'], reward)

    def __get_done_reward(self):
        winners = self.blokus_game.winners()
        done = winners is not None
        if done:
            if len(winners) == 1 and winners[0] == self.starter_player:
                self.starter_won += 1
            if "ai" in winners:
                if len(winners) == 1:
                    reward = self.rewards['won']
                else:
                    reward = self.rewards['tie-won']
            else:
                reward = self.rewards['lost']
        else:
            reward = self.rewards['default'] if self.ai.next_move is None else self.ai.next_move.size
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
            dummy = Player("", "", self.all_shapes, self.blokus_game)

            # self.all_possible_indexes_to_moves = possible_moves_func(dummy, self.BOARD_SIZE, self.all_shapes)
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
