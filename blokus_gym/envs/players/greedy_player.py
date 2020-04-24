import numpy as np
from blokus_gym.envs.players.player import Player


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

        best_moves = []
        best_score = self.score_move(possible_moves[0])
        for move in possible_moves:
            new_score = self.score_move(move)
            if new_score > best_score:
                best_score = new_score
                best_moves = [move]
            elif new_score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)
