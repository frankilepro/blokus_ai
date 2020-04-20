import matplotlib.pyplot as plt
import os
import numpy as np
import torch


class Board:
    """
    Creates a board that has n rows and
    m columns with an empty space represented
    by a character string according to null of
    character length one.
    """

    def __init__(self, size):
        plt.ion()
        self.size = size
        self.player_ids = {}
        self.tensor = torch.zeros((size, size), dtype=torch.int32)
        # self.null = null
        # self.empty = [[self.null] * m for i in range(n)]
        # self.state = self.empty

    def update(self, player, move):
        """
        Takes in a Player object and a move as a
        list of integer tuples that represent the piece.
        """
        if player.index not in self.player_ids:
            self.player_ids[player.index] = len(self.player_ids) + 1  # since 0 represents empty

        for x, y in move.points:
            self.tensor[y][x] = player.index
        # for row in range(self.size):
        #     for col in range(self.size):
        #         if (col, row) in move:
        #             # self.state[row][col] = player.index
        #             self.tensor[row][col] = player.index  # TODO optimize with single container

    def in_bounds(self, point):
        """
        Takes in a tuple and checks if it is in the bounds of
        the board.
        """
        x, y = point
        return 0 <= x < self.size and 0 <= y < self.size

    def overlap(self, points):
        """
        Returns a boolean for whether a move is overlapping
        any pieces that have already been placed on the board.
        """
        return any(self.tensor[y][x] != 0 for x, y in points)
        # if False in [(self.tensor[j][i] == 0) for (i, j) in move]:
        #     return(True)
        # else:
        #     return(False)

    def corner(self, player, move):
        """
        Note: ONLY once a move has been checked for adjacency, this
        function returns a boolean; whether the move is cornering
        any pieces of the player proposing the move.
        """
        for x, y in move.points:
            if self.in_bounds((x + 1, y + 1)) and self.tensor[y + 1][x + 1] == player.index or \
               self.in_bounds((x - 1, y - 1)) and self.tensor[y - 1][x - 1] == player.index or \
               self.in_bounds((x - 1, y + 1)) and self.tensor[y + 1][x - 1] == player.index or \
               self.in_bounds((x + 1, y - 1)) and self.tensor[y - 1][x + 1] == player.index:
                return True
        return False
        # validates = []
        # for (i, j) in move.points:
        #     if self.in_bounds((j + 1, i + 1)):
        #         validates.append((self.tensor[j + 1][i + 1] == player.index))
        #     if self.in_bounds((j - 1, i - 1)):
        #         validates.append((self.tensor[j - 1][i - 1] == player.index))
        #     if self.in_bounds((j - 1, i + 1)):
        #         validates.append((self.tensor[j - 1][i + 1] == player.index))
        #     if self.in_bounds((j + 1, i - 1)):
        #         validates.append((self.tensor[j + 1][i - 1] == player.index))
        # if True in validates:
        #     return True
        # else:
        #     return False

    def adj(self, player, move):
        """
        Checks if a move is adjacent to any squares on
        the board which are occupied by the player
        proposing the move and returns a boolean.
        """
        for x, y in move.points:
            if self.in_bounds((x, y + 1)) and self.tensor[y + 1][x].item() == player.index or \
               self.in_bounds((x, y - 1)) and self.tensor[y - 1][x].item() == player.index or \
               self.in_bounds((x - 1, y)) and self.tensor[y][x - 1].item() == player.index or \
               self.in_bounds((x + 1, y)) and self.tensor[y][x + 1].item() == player.index:
                return True
        return False
        # validates = []
        # for (i, j) in move.points:
        #     if self.in_bounds((j, i + 1)):
        #         validates.append((self.tensor[j][i + 1].item() == player.index))
        #     if self.in_bounds((j, i - 1)):
        #         validates.append((self.tensor[j][i - 1].item() == player.index))
        #     if self.in_bounds((j - 1, i)):
        #         validates.append((self.tensor[j - 1][i].item() == player.index))
        #     if self.in_bounds((j + 1, i)):
        #         validates.append((self.tensor[j + 1][i].item() == player.index))
        # if True in validates:
        #     return True
        # else:
        #     return False

    def print_board(self, num=None, mode="human"):
        if mode == "human":
            self.fancyBoard(num)
        elif mode == "minimal":
            self.print_board_min()
        elif mode == "tensor":
            print(self.tensor)
        elif mode == "old":
            self.printBoard()

    def printBoard(self):
        n = 2
        """
        Prints the board where the representation of a board is
        a list of row-lists. The function throws an error if the
        the board is invalid: the length of the rows are not
        the same.
        """
        assert(len(set([len(self.tensor[i]) for i in range(len(self.tensor))])) == 1)
        print(' ' * n, end=' ')
        for i in range(len(self.tensor[1])):
            print(str(i) + ' ' * (n-len(str(i))), end=' ')
        print()
        for i, row in enumerate(self.tensor):
            print(str(i) + ' ' * (n-len(str(i))), (' ' * n).join(row))

    def print_board_min(self):
        all_non_zeros = self._tensor != 0
        coverage = all_non_zeros.sum().item() / (all_non_zeros.shape[0] * all_non_zeros.shape[1]) * 100
        print(f"Coverage: {coverage:.2f}%")

    def fancyBoard(self, num):
        plt.clf()
        # points = {}
        # for y in enumerate(self.tensor):
        #     for x in enumerate(y[1]):
        #         id = x[1].item()
        #         if id not in points:
        #             points[id] = []
        #         points[id].append((x[0], (self.size - 1) - y[0]))

        colors = {0: "lightgrey", 1: "red", 2: "blue", 3: "yellow", 4: "green"}
        ax = plt.subplot(xlim=(0, self.size), ylim=(0, self.size))
        for y in range(self.size):
            for x in range(self.size):
                polygon = plt.Polygon([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
                polygon.set_facecolor(colors[self.tensor[y][x].item()])
                ax.add_patch(polygon)
        # for i in range(self.size + 1):
        #     for j in range(self.size + 1):
        #         polygon = plt.Polygon([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1], [i, j]])
        #         for player, pts in points.items():
        #             if (i, j) in pts:
        #                 polygon.set_facecolor(colors[player])
        #                 ax.add_patch(polygon)

        plt.yticks(np.arange(0, self.size, 1))
        plt.xticks(np.arange(0, self.size, 1))
        plt.grid()
        plt.draw()
        plt.pause(0.00001)
