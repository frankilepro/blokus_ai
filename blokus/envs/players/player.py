import copy
import numpy as np
import random


def eval_move(piece, player, game, weights):
    """
    Takes in a single Piece object and a Player object and returns a integer score that
    evaluates how "good" the Piece move is. Defined here because used by both Greedy and Minimax.
    """
    def check_corners(player):
        """
        Updates the corners of the player in the test board (copy), in case the
        corners have been covered by another player's pieces.
        """
        player.corners = set([(i, j) for (i, j) in player.corners if test_board.tensor[j][i] == 0])
    # get board
    board = game.board
    # create a copy of the players in the game
    test_players = copy.deepcopy(game.players)
    # create a list of the opponents in the game
    opponents = [opponent for opponent in test_players if opponent.index != player.index]
    # create a copy of the board
    test_board = copy.deepcopy(board)
    # update the copy of the board with the Piece placement
    test_board.update(player, piece)
    # create a copy of the player currently playing
    test_player = copy.deepcopy(player)
    # update the current player (update corners) with the current Piece placement
    test_player.update_player(piece, test_board)
    # calculate how many corners the current player has
    my_corners = len(test_player.corners)
    # update the corners for all opponents
    list(map(check_corners, opponents))
    # calculate the mean of the corners of the opponents
    opponent_corners = [len(opponent.corners) for opponent in opponents]
    # find the difference between the number of corners the current player has and and the
    # mean number of corners the opponents have
    corner_difference = np.mean([my_corners - opponent_corner for opponent_corner in opponent_corners])
    # return the score = size + difference in the number of corners
    return (piece, weights[0] * piece.size + weights[1] * corner_difference)


class Player:
    def __init__(self, index, name, all_moves, game, deterministic=False):
        self.index = index
        self.name = name
        self.corners = set()
        self.score = 0
        self.game = game
        self.rng = random.Random(0) if deterministic else random
        self.__set_all_ids_to_move(all_moves)

    def __set_all_ids_to_move(self, all_moves):
        self.all_ids_to_move = {}
        for move in all_moves:
            if move.ID not in self.all_ids_to_move:
                self.all_ids_to_move[move.ID] = []
            self.all_ids_to_move[move.ID].append(move)

        # For performance issue, we shuffle first, then we look "in order" to find the first move
        # in casses when we want to sample a random move
        for key in self.all_ids_to_move.keys():
            self.rng.shuffle(self.all_ids_to_move[key])

    def add_pieces(self, pieces):
        """
        Gives a player the initial set of pieces.
        """
        piece_ids = set(p.ID for p in pieces)
        for missing_piece_id in self.all_ids_to_move.keys() - piece_ids:
            self.all_ids_to_move.remove(missing_piece_id)

    def start_corner(self, p):
        """
        Gives a player an initial starting corner.
        """
        self.corners = set([p])

    def remove_piece(self, piece):
        """
        Removes a given piece (Shape object) from
        the list of pieces a player has.
        """
        del self.all_ids_to_move[piece.ID]

    def update_player(self, placement, board):
        """
        Updates the variables that the player is keeping track
        of, e.g. their score and their available corners.
        Placement should be in the form of a Shape object.
        """
        self.score = self.score + placement.size
        for c in placement.corners:
            if board.in_bounds(c) and not board.overlap([c]):
                self.corners.add(c)

    def sample_move(self):
        keys = list(self.all_ids_to_move.keys())
        self.rng.shuffle(keys)
        nb = 0
        for key in keys:
            nb += 1
            for move in self.all_ids_to_move[key]:
                if self.game.valid_move(self, move):
                    return move
        return None

    def sample_move_idx(self):
        move = self.sample_move()
        return None if move is None else move.idx

    @property
    def remains_move(self):
        for moves in self.all_ids_to_move.values():
            if any(self.game.valid_move(self, move) for move in moves):
                return True
        return False

    def possible_moves_opt(self):
        placements = []
        for moves in self.all_ids_to_move.values():
            placements.extend(move for move in moves if self.game.valid_move(self, move))
        return placements

    def possible_move_indexes(self):
        return [move.idx for move in self.possible_moves_opt()]

    def possible_moves(self, pieces, no_restriction=False, board_size=14):
        """
        Returns a unique list of placements, i.e. Shape objects
        with a particular flip, orientation, corners, and points.
        It uses a list of pieces (Shape objects) and the game, which includes
        its rules and valid moves, in order to find the placements.
        """
        def check_corners(no_restriction):
            """
            Updates the corners of the player, in case the
            corners have been covered by another player's pieces.
            """
            if no_restriction:
                self.corners = set()
                for i in range(-6, board_size + 6):
                    for j in range(-6, board_size + 6):
                        self.corners.add((i, j))
            else:
                self.corners = set([(i, j) for (i, j) in self.corners if self.game.board.tensor[j][i] == 0])

        # Check the corners before proceeding.
        check_corners(no_restriction)
        # This list of placements will be updated with valid ones.
        placements = set()
        # Loop through every available corner.
        for x, y in self.corners:
            # Look through every piece offered. (This will be restricted according
            # to certain algorithms.)
            for sh in pieces:
                # Create a new shape so that the one in the player's
                # list of shapes is not overwritten.
                try_out = copy.deepcopy(sh)
                try_out.set_points(x, y)
                # And every possible flip.
                for _ in range(2):
                    try_out.flip()
                    # And every possible orientation.
                    for _ in range(4):
                        try_out.rotate()
                        candidate = copy.deepcopy(try_out)
                        if self.game.valid_move(self, candidate):
                            placements.add(candidate)
        return list(placements)

    def do_move(self):
        """
        Generates a move according to the Player's
        strategy and current state of the board.
        """
        return None
