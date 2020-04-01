# Inspired from https://github.com/mknapper1/Machine-Learning-Blokus/
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
from matplotlib import rcParams
from player import Player
from random_player import Random_Player
from board import Board
from blokus_game import BlokusGame
import shapes
rcParams['figure.figsize'] = (6, 6)
rcParams['figure.dpi'] = 150


All_Shapes = [shapes.I1(), shapes.I2(), shapes.I3(), shapes.I4(), shapes.I5(),
              shapes.V3(), shapes.L4(), shapes.Z4(), shapes.O4(), shapes.L5(),
              shapes.T5(), shapes.V5(), shapes.N(), shapes.Z5(), shapes.T4(),
              shapes.P(), shapes.W(), shapes.U(), shapes.F(), shapes.X(), shapes.Y()]
All_Degrees = [0, 90, 180, 270]
All_Flip = ['h', "None"]


print("\n \n Welcome to Blokus! \n \n \n Blokus is a geometrically abstract, strategy board game. It can be a two- or four-player game. Each player has 21 pieces of a different color. The two-player version of the board has 14 rows and 14 columns. \n \n You will be playing a two-player version against an algorithm of your choice: Random, Greedy, or Minimax. In case you need to review the rules of Blokus, please follow this link: http://en.wikipedia.org/wiki/Blokus. \n \n This is how choosing a move is going to work: after every turn, we will display the current state of the board, as well as the scores of each player and the pieces available to you. We have provided you with a map of the names of the pieces, as well as their reference points, denoted by red dots. When you would like to place a piece, we will prompt you for the name of the piece and the coordinate (column, row) of the reference point. If multiple placements are possible, we will let you choose which one you would like to play. \n \n Good luck! \n \n")
img = Image.open('tiles.png')
img.show()
# print "Please choose an algorithm to play against: \n A. Random \n B. Greedy \n C. Minimax \n"
# choice = raw_input().upper()
# while not (choice in ["A", "B", "C"]):
#    choice = raw_input("\n Please choose a valid algorithm: \n").upper()
#
# if choice == "A":
#    computer = Player("A", "Computer", Random_Player)
# elif choice == "B":
#    computer = Greedy("A", "Computer", Greedy_Player, [2, 1, 5, 1, 1])
# else:
#    computer = Greedy("A", "Computer", Minimax_Player, [2, 1, 5, 1, 1])
first = Player("A", "Computer_A", Random_Player)
second = Player("B", "Computer_B", Random_Player)
third = Player("C", "Computer_C", Random_Player)
fourth = Player("D", "Computer_D", Random_Player)
standard_size = Board(14, 14, "_")
ordering = [first, second, third, fourth]
random.shuffle(ordering)
userblokus = BlokusGame(ordering, standard_size, All_Shapes)
userblokus.board.print_board(num=userblokus.rounds, fancy=False)
print("\n")
userblokus.play()
userblokus.board.print_board(num=userblokus.rounds, fancy=False)
print("\n")
while userblokus.winner() == "None":
    userblokus.play()
    userblokus.board.print_board(num=userblokus.rounds, fancy=False)
    print("\n")
    for p in userblokus.players:
        print(p.name + " (" + str(p.score) + ") : " + str([s.ID for s in p.pieces]))
        print()
    print("=======================================================================")
print()
userblokus.board.print_board()
print()
userblokus.play()
print("The final scores are...")
by_name = sorted(userblokus.players, key=lambda player: player.name)
for p in by_name:
    print(p.name + " : " + str(p.score))
