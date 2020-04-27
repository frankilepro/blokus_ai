from sys import maxsize
import copy
from multiprocessing import Pool
import os
import numpy as np

from blokus_gym.envs.players.player import Player


def iterate_over_moves(player, depth, possible_moves, prev_moves):
    score_move = [None, [- maxsize - 1] * len(player.game.players)]
    for move in possible_moves:
        node = copy.deepcopy(player.game)
        MinimaxPlayer.play_without_do_move(node, move)
        current_score = MinimaxPlayer.minimax(node, depth - 1, depth, prev_moves + [move])
        score_move = max(score_move, current_score, key=lambda x: x[1][player.index - 1])
    return score_move


class MinimaxPlayer(Player):

    @staticmethod
    def score_players(game):
        scores = np.zeros(len(game.players))
        for player in game.players:
            scores[player.index - 1] = len(player.corners) + 5 * player.score
        return 2 * scores - np.sum(scores)

    @staticmethod
    def play_without_do_move(game, move):
        if game.winners() is None:
            current = game.players[0]
            # print("Current player: " + current.name)
            proposal = move
            if proposal is None:
                # move on to next player, increment rounds
                first = game.players.pop(0)
                game.players = game.players + [first]
                game.rounds += 1
            # ensure that the proposed move is valid
            elif game.valid_move(current, proposal):
                # update the board with the move
                game.board.update(current, proposal)
                # let the player update itself accordingly
                current.update_player(proposal, game.board)
                # remove the piece that was played from the player
                current.remove_piece(proposal)
                # place the player at the back of the queue
                first = game.players.pop(0)
                game.players = game.players + [first]
                # increment the number of rounds just played
                game.rounds += 1

    @staticmethod
    def minimax(game, depth, start_depth, prev_moves):
        player = game.next_player()
        possible_moves = player.possible_moves_opt()
        np.random.shuffle(possible_moves)
        nb_possible_moves = len(possible_moves)
        if depth < 0 or nb_possible_moves == 0:
            return [prev_moves, MinimaxPlayer.score_players(game)]

        if depth == start_depth:
            args = [(player, depth, [move], prev_moves) for move in possible_moves]
            with Pool(os.cpu_count()) as pool:
                scores = pool.starmap(iterate_over_moves, args)
                max_score = max(scores, key=lambda x: x[1][player.index - 1])
                return max_score

        return iterate_over_moves(player, depth, possible_moves, prev_moves)

    def do_move(self):
        moves = MinimaxPlayer.minimax(copy.deepcopy(self.game), 1, 1, [])[0]
        return moves[0] if moves else None
