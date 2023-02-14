import matplotlib.pyplot as plt
import numpy as np
import torch
import pygame
from pygame.locals import *


class Board:
    """
    Creates a board that has n rows and
    m columns with an empty space represented
    by a character string according to null of
    character length one.
    """

    def __init__(self, size, window=None):
        plt.ion()
        self.size = size
        self.tensor = torch.zeros((size, size), dtype=torch.int32)

        if window is not None:
            self.window = window
            self.colors = {0: (211, 211, 211), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (0, 255, 0)}
            self.clock = pygame.time.Clock()
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT = self.window.get_size()

    def update(self, player, move):
        """
        Takes in a Player object and a move as a
        list of integer tuples that represent the piece.
        """
        for x, y in move.points:
            self.tensor[y][x] = player.index

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
        return any(self.tensor[y][x].item() != 0 for x, y in points)

    def is_player_tile(self, player, point):
        x, y = point
        return self.in_bounds((x, y)) and self.tensor[y][x].item() == player.index

    def corner(self, player, move):
        """
        Note: ONLY once a move has been checked for adjacency, this
        function returns a boolean; whether the move is cornering
        any pieces of the player proposing the move.
        """
        return any(self.is_player_tile(player, (x + 1, y + 1)) or self.is_player_tile(player, (x - 1, y - 1)) or
                   self.is_player_tile(player, (x - 1, y + 1)) or self.is_player_tile(player, (x + 1, y - 1))
                   for x, y in move.points)

    def adj(self, player, move):
        """
        Checks if a move is adjacent to any squares on
        the board which are occupied by the player
        proposing the move and returns a boolean.
        """
        return any(self.is_player_tile(player, (x, y + 1)) or self.is_player_tile(player, (x, y - 1)) or
                   self.is_player_tile(player, (x - 1, y)) or self.is_player_tile(player, (x + 1, y))
                   for x, y in move.points)

    def print_board(self, render_mode="human"):
        if render_mode == "human":
            self.pygame_board()
        elif render_mode == "pyplot":
            self.pyplot_board()
        elif render_mode == "minimal":
            self.print_board_min()
        elif render_mode == "tensor":
            self.print_tensor()


    def print_tensor(self):
        print(chr(27) + "[2J")
        print(self.tensor.permute())

    def print_board_min(self):
        print(chr(27) + "[2J")
        all_non_zeros = self.tensor != 0
        coverage = all_non_zeros.sum().item() / (all_non_zeros.shape[0] * all_non_zeros.shape[1]) * 100
        print(f"Coverage: {coverage:.2f}%")

    def pyplot_board(self):
        plt.clf()

        colors = {0: "lightgrey", 1: "red", 2: "blue", 3: "yellow", 4: "green"}
        ax = plt.subplot(xlim=(0, self.size), ylim=(0, self.size))
        for y in range(self.size):
            for x in range(self.size):
                polygon = plt.Polygon([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
                polygon.set_facecolor(colors[self.tensor[y][x].item()])
                ax.add_patch(polygon)

        plt.yticks(np.arange(0, self.size, 1))
        plt.xticks(np.arange(0, self.size, 1))
        plt.grid()
        plt.draw()
        plt.pause(0.00001)

    def pygame_board(self):
        self.clock.tick(1)

        # Only render if there is something to render
        blockSize = int(self.WINDOW_WIDTH / self.size) # Set the size of the grid block
        for x, x_screen in enumerate(range(0, self.WINDOW_WIDTH, blockSize)):
            for y, y_screen in enumerate(range(0, self.WINDOW_HEIGHT, blockSize)):
                rect = pygame.Rect(x_screen, y_screen, blockSize, blockSize)
                pygame.draw.rect(self.window, self.colors[self.tensor[y][x].item()], rect, 0) # Fill
                pygame.draw.rect(self.window, (150, 150, 150), rect, 1) # Border

        pygame.display.update()
