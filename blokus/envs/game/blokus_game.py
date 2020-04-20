from blokus.envs.game.game import Game


class BlokusGame(Game):
    """
    A class that takes a list of players, e.g. ['A','B','C'],
    and a board and plays moves, keeping track of the number
    of rounds that have been played.
    """

    def winners(self):
        """
        Checks the conditions of the game
        to see if the game has been won yet
        and returns None if the game has
        not been won, and the name of the
        player if it has been won.
        """
        if any(p.remains_move for p in self.players):
            return None
        else:
            winners = []
            winner_score = 0
            for player in self.players:
                if player.score > winner_score:
                    winners = [player.name]
                    winner_score = winner_score
                elif player.score == winner_score:
                    winners.append(player.name)
            return winners

    def valid_move(self, player, move):
        """
        Uses functions from the board to see whether
        a player's proposed move is valid.
        """
        # if move.ID not in player.all_ids_to_move:
        #     return False
        # if player.game.rounds == 10 and move.ID == 'L4' and player.index == 1:
        # if sorted([(3, 2), (4, 2), (5, 2), (5, 3)]) == sorted(move.points):
        #     print("", end="")

        if any(not self.board.in_bounds(pt) for pt in move.points):
            return False
        if self.board.overlap(move.points):
            return False

        if self.rounds < self.number_of_players:
            if not any(pt in player.corners for pt in move.points):
                return False
        else:
            if self.board.adj(player, move) or not self.board.corner(player, move):
                return False

        return True
