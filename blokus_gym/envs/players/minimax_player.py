import numpy as np
from sys import maxsize
from blokus_gym.envs.players.player import Player
import copy


class MinimaxPlayer(Player):
    @staticmethod
    def possible_moves_of_size(player, size):
        placements = []
        for moves in player.all_labels_to_move.values():
            placements.extend(move for move in moves if move.size == size and player.game.valid_move(player, move))
        return placements

    @staticmethod
    def possible_moves_bellow_size(player, size):
        possible_moves = []
        while len(possible_moves) == 0:
            possible_moves = MinimaxPlayer.possible_moves_of_size(player, size)
            size -= 1
            if size == 0 and len(possible_moves) == 0:
                return None
        return possible_moves

    @staticmethod
    def score_player(player):
        self_corners_len = len(player.corners)
        # TODO opt set
        len_differentials = [self_corners_len - (len(game_player.corners) - len(game_player.corners))
                             for game_player in player.game.players if game_player != player]
        return np.mean(len_differentials)

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

    def minimax(self, game, depth):
        size = 5
        player = game.next_player()
        possible_moves = MinimaxPlayer.possible_moves_bellow_size(player, size)
        if depth == 0 or len(possible_moves) == 0:
            return (None, self.score_player(player))

        if player is self:
            move_score = (None, - maxsize - 1)
            for move in possible_moves:
                node = copy.deepcopy(self.game)
                MinimaxPlayer.play_without_do_move(node, move)
                move_score = max(move_score, self.minimax(node, depth - 1), key=lambda x: x[1])
            return move_score
        else:
            move_score = (None, maxsize)
            for move in possible_moves:
                node = copy.deepcopy(self.game)
                MinimaxPlayer.play_without_do_move(node, move)
                move_score = min(move_score, self.minimax(node, depth - 1), key=lambda x: x[1])
            return move_score

    def do_move(self):
        return self.minimax(copy.deepcopy(self.game), 1)[0]
