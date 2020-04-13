import matplotlib.pyplot as plt
import os
import numpy as np


class Board:
    """
    Creates a board that has n rows and
    m columns with an empty space represented
    by a character string according to null of
    character length one.
    """

    def __init__(self, n, m, null):
        plt.ion()
        self.size = (n, m)
        self.player_ids = {}
        self.np_board = np.zeros(self.size, dtype=int)
        self.null = null
        self.empty = [[self.null] * m for i in range(n)]
        self.state = self.empty

    def numpy(self):
        return self.np_board

    def update(self, player, move):
        """
        Takes in a Player object and a move as a
        list of integer tuples that represent the piece.
        """
        if player.label not in self.player_ids:
            self.player_ids[player.label] = len(self.player_ids) + 1  # since 0 represents empty

        id = self.player_ids[player.label]
        for row in range(len(self.state)):
            for col in range(len(self.state[1])):
                if (col, row) in move:
                    self.state[row][col] = player.label
                    self.np_board[row][col] = id

    def in_bounds(self, point):
        """
        Takes in a tuple and checks if it is in the bounds of
        the board.
        """
        return (0 <= point[0] <= (self.size[1] - 1)) & (0 <= point[1] <= (self.size[0] - 1))

    def overlap(self, move):
        """
        Returns a boolean for whether a move is overlapping
        any pieces that have already been placed on the board.
        """
        if False in [(self.state[j][i] == self.null) for (i, j) in move]:
            return(True)
        else:
            return(False)

    def corner(self, player, move):
        """
        Note: ONLY once a move has been checked for adjacency, this
        function returns a boolean; whether the move is cornering
        any pieces of the player proposing the move.
        """
        validates = []
        for (i, j) in move:
            if self.in_bounds((j + 1, i + 1)):
                validates.append((self.state[j + 1][i + 1] == player.label))
            if self.in_bounds((j - 1, i - 1)):
                validates.append((self.state[j - 1][i - 1] == player.label))
            if self.in_bounds((j - 1, i + 1)):
                validates.append((self.state[j - 1][i + 1] == player.label))
            if self.in_bounds((j + 1, i - 1)):
                validates.append((self.state[j + 1][i - 1] == player.label))
        if True in validates:
            return True
        else:
            return False

    def adj(self, player, move):
        """
        Checks if a move is adjacent to any squares on
        the board which are occupied by the player
        proposing the move and returns a boolean.
        """
        validates = []
        for (i, j) in move:
            if self.in_bounds((j, i + 1)):
                validates.append((self.state[j][i + 1] == player.label))
            if self.in_bounds((j, i - 1)):
                validates.append((self.state[j][i - 1] == player.label))
            if self.in_bounds((j - 1, i)):
                validates.append((self.state[j - 1][i] == player.label))
            if self.in_bounds((j + 1, i)):
                validates.append((self.state[j + 1][i] == player.label))
        if True in validates:
            return True
        else:
            return False

    def print_board(self, num=None, fancy=True):
        if fancy:
            self.fancyBoard(num)
        else:
            self.printBoard()

    def printBoard(self):
        n = 2
        """
        Prints the board where the representation of a board is
        a list of row-lists. The function throws an error if the
        the board is invalid: the length of the rows are not
        the same.
        """
        assert(len(set([len(self.state[i]) for i in range(len(self.state))])) == 1)
        print(' ' * n, end=' ')
        for i in range(len(self.state[1])):
            print(str(i) + ' ' * (n-len(str(i))), end=' ')
        print()
        for i, row in enumerate(self.state):
            print(str(i) + ' ' * (n-len(str(i))), (' ' * n).join(row))

    def fancyBoard(self, num):
        plt.clf()
        points = {}
        for y in enumerate(self.state):
            for x in enumerate(y[1]):
                id = x[1]
                if id not in points:
                    points[id] = []
                points[id].append((x[0], (self.size[0] - 1) - y[0]))

        colors = {"A": "red", "B": "blue", "C": "yellow", "D": "green", "_": "lightgrey"}
        ax = plt.subplot(xlim=(0, self.size[0]), ylim=(0, self.size[1]))
        for i in range(self.size[0] + 1):
            for j in range(self.size[1] + 1):
                polygon = plt.Polygon([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1], [i, j]])
                for player, pts in points.items():
                    if (i, j) in pts:
                        polygon.set_facecolor(colors[player])
                        ax.add_patch(polygon)

        plt.xticks(np.arange(0, self.size[0], 1))
        plt.yticks(np.arange(0, self.size[1], 1))
        plt.grid()
        plt.draw()
        plt.pause(0.00001)
