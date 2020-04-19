from blokus.envs.game.game import Game


class BlokusGame(Game):
    """
    A class that takes a list of players, e.g. ['A','B','C'],
    and a board and plays moves, keeping track of the number
    of rounds that have been played.
    """

    def winner(self):
        """
        Checks the conditions of the game
        to see if the game has been won yet
        and returns None if the game has
        not been won, and the name of the
        player if it has been won.
        """
        # Credit to Dariusz Walczak for inspiration.
        # http://stackoverflow.com/questions/1720421/merge-two-lists-in-python
        remains_moves = [p.remains_move for p in self.players]
        if True in remains_moves:
            return None
        else:
            cand = [(p.score, p.name) for p in self.players]
            return sorted(cand, reverse=True)[0][1]

    def valid_move(self, player, move):
        """
        Uses functions from the board to see whether
        a player's proposed move is valid.
        """
        if move.ID not in player.all_ids_to_move:
            return False

        move_points = move.points
        if any(not self.board.in_bounds(pt) for pt in move_points):
            return False
        if self.board.overlap(move_points):
            return False

        if self.rounds < self.number_of_players:
            if any(pt not in player.corners for pt in move_points):
                return False
        else:
            if self.board.adj(player, move_points) or not self.board.corner(player, move_points):
                return False

        return True
