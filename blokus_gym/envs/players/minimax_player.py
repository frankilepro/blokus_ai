import numpy as np
from sys import maxsize
from blokus_gym.envs.players.player import Player
import copy
from multiprocessing import Pool, Lock
import os


def iterate_over_moves(player, depth, possible_moves, prev_moves):
    score_move = [None, [- maxsize - 1] * len(player.game.players)]
    for move in possible_moves:
        MinimaxPlayer.play_without_do_move(player.game, move)
        current_score = MinimaxPlayer.minimax(player.game, depth - 1, depth, prev_moves + [move])
        MinimaxPlayer.undo_play_without_do_move(player.game, move)
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
    def undo_play_without_do_move(game, move):
        last = game.players[-1]
        proposal = move
        # update the board with the move
        game.board.undo_update(last, proposal)
        # let the player update itself accordingly
        last.undo_update_player(proposal)
        # remove the piece that was played from the player
        last.undo_remove_piece()
        # place the player at the back of the queue
        last = game.players.pop()
        game.players = [last] + game.players
        game.rounds -= 1

    @staticmethod
    def minimax(game, depth, start_depth, prev_moves):
        player = game.next_player()
        possible_moves = [move for move in player.possible_moves_opt() if game.valid_move(player, move)]
        nb_possible_moves = len(possible_moves)
        if depth < 0 or nb_possible_moves == 0:
            return [prev_moves, MinimaxPlayer.score_players(game)]

        if depth == start_depth and nb_possible_moves >= os.cpu_count():
            process_split_factor = nb_possible_moves // os.cpu_count()
            split_possible_moves = np.split(possible_moves, list(range(0, nb_possible_moves, process_split_factor)))[1:]
            args = [(player, depth, moves, prev_moves) for moves in split_possible_moves]
            with Pool(os.cpu_count()) as pool:
                scores = (pool.starmap(iterate_over_moves, args))
                max_score = max(scores, key=lambda x: x[1][player.index - 1])
                return max_score

        return iterate_over_moves(player, depth, possible_moves, prev_moves)

    def do_move(self):
        moves = MinimaxPlayer.minimax(copy.deepcopy(self.game), 1, -1, [])[0]
        return moves[0] if moves else None
