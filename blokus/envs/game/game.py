class Game:
    """
    A class that takes a list of players objects,
    and a board object and plays moves, keeping track of the number
    of rounds that have been played and determining the validity
    of moves proposed to the game.
    """

    def __init__(self, players, board, all_pieces):
        self.players = players
        self.rounds = 0
        self.board = board
        self.all_pieces = all_pieces

    def winner(self):
        """
        Checks the conditions of the game
        to see if the game has been won yet
        and returns "None" if the game has
        not been won, and the name of the
        player if it has been won.
        """
        return("None")

    def valid_move(self, player, move):
        """
        Uses functions from the board to see whether
        a player's proposed move is valid.
        """
        return(True)

    def play(self):
        """
        Plays a list of Player objects sequentially,
        as long as the game has not been won yet,
        starting with the first player in the list at
        instantiation.
        """
        if self.rounds == 0:
            # When the game has not begun yet, the game must
            # give the players their pieces and a corner to start.
            max_x = ((self.board).size[1] - 1)
            max_y = ((self.board).size[0] - 1)
            starts = [(0, 0), (max_y, max_x), (0, max_x), (max_y, 0)]
            for i in range(len(self.players)):
                (self.players[i]).add_pieces(self.all_pieces)
                (self.players[i]).start_corner(starts[i])
        # if there is no winner, print out the current player's name and
        # let current player perform a move
        if self.winner() == "None":
            current = self.players[0]
            print("Current player: " + current.name)
            proposal = current.do_move(self)
            if proposal == None:
                # move on to next player, increment rounds
                first = (self.players).pop(0)
                self.players = self.players + [first]
                self.rounds += 1
            # ensure that the proposed move is valid
            elif self.valid_move(current, proposal.points):
                # update the board with the move
                (self.board).update(current, proposal.points)
                # let the player update itself accordingly
                current.update_player(proposal, self.board)
                # remove the piece that was played from the player
                current.remove_piece(proposal)
                # place the player at the back of the queue
                first = (self.players).pop(0)
                self.players = self.players + [first]
                # increment the number of rounds just played
                self.rounds += 1
            # interrupts the game if an invalid move is proposed
            else:
                raise Exception("Invalid move by " + current.name + ".")
        else:
            print("Game over! And the winner is: " + self.winner())
