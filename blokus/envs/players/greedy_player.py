from blokus.envs.players.player import Player
import numpy as np

# def Greedy_Player(player, game, weights):
#     """
#     Takes in a Player object and Game object and returns a placement in the form of a
#     single piece object with a proper flip, orientation, corners, and points.
#     If no placement can be made, function should return None.
#     """
#     # create copy of player's pieces (no destructively altering player's pieces)
#     shape_options = [p for p in player.pieces]
#     board = game.board

#     def greedy_move():
#         """
#         Returns the greediest move.
#         """
#         # create an empty list that will contain all the possible moves with their respective scores
#         final_moves = []
#         # for each piece, calculate all possible placements, and for each placement, calculate the score
#         # of the move; add (move, score) to the list of final moves
#         for piece in shape_options:
#             # calculate all possible placements for the current piece
#             possibles = player.possible_moves([piece], game)
#             # if there are possible placements for the current piece:
#             if possibles != []:
#                 def map_eval(piece):
#                     return eval_move(piece, player, game, weights)
#                 # calculate score for each move and store it in a temporary list
#                 tmp = list(map(map_eval, possibles))
#                 # add all the elements in the temporary list in the final moves lsit
#                 final_moves.extend(tmp)
#             # if there are no possible placements for the current piece:
#             else:
#                 # remove the piece from the list of pieces
#                 shape_options.remove(piece)
#         # create score list that contains all Piece placements, sorted by their score
#         by_score = sorted(final_moves, key=lambda move: move[1], reverse=True)
#         # if the score list contains Piece placements (objects), return the highest scoring Piece placement
#         if len(by_score) > 0:
#             return by_score[0][0]
#         # else, return None (no Piece placement)
#         else:
#             return None
#     # while there are shapes to place down, perform a greedy move
#     return greedy_move()


class GreedyPlayer(Player):
    def possible_moves_of_size(self, size):
        placements = []
        for moves in self.all_ids_to_move.values():
            placements.extend(move for move in moves if move.size == size and self.game.valid_move(self, move))
        return placements

    def score_move(self, move):
        self_corners_len = len(self.corners) + len(move.corners)
        # TODO opt set
        len_differentials = [self_corners_len - (len(player.corners) - len(player.corners - set(move.corners)))
                             for player in self.game.players if player != self]
        return np.mean(len_differentials)

    def do_move(self):
        # This is more optimal than to look at all possible moves,
        # it's also more greedy, because we always look size first
        size = 5
        possible_moves = []
        while len(possible_moves) == 0:
            possible_moves = self.possible_moves_of_size(size)
            size -= 1
            if size == 0 and len(possible_moves) == 0:
                return None

        best_move = possible_moves[0]
        best_score = self.score_move(best_move)
        for move in possible_moves:
            new_score = self.score_move(move)
            if new_score > best_score:
                best_score = new_score
                best_move = move

        return best_move
