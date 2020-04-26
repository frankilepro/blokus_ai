import numpy as np
from sys import maxsize
from blokus_gym.envs.players.player import Player
import copy


class MinimaxPlayer(Player):
    @staticmethod
    def score_player(player):
        self_corners_len = len(player.corners)
        # TODO opt set
        len_differentials = [self_corners_len - (len(game_player.corners) - len(game_player.corners))
                             for game_player in player.game.players if game_player != player]

        score_differentials = [player.score - (game_player.score - game_player.score)
                               for game_player in player.game.players if game_player != player]
        return np.mean(len_differentials) + 5 * np.mean(score_differentials)

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

    def minimax(self, game, depth, prev_move):
        player = game.next_player()
        possible_moves = player.possible_moves_opt()
        if depth < 0 or len(possible_moves) == 0:
            return (self.score_player(player), prev_move)

        if player.index == self.index:
            score_move = (- maxsize - 1, None)
            for move in possible_moves:
                node = copy.deepcopy(player.game)
                MinimaxPlayer.play_without_do_move(node, move)
                score_move = max(score_move, self.minimax(node, depth - 1, move))
            return score_move
        else:
            score_move = (maxsize, None)
            for move in possible_moves:
                node = copy.deepcopy(player.game)
                MinimaxPlayer.play_without_do_move(node, move)
                score_move = min(score_move, self.minimax(node, depth - 1, move))
            return score_move

    def do_move(self):
        score_move = self.minimax(copy.deepcopy(self.game), 2, None)

        return score_move[1]
