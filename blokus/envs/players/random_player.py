import random


def Random_Player(player, game):
    """
    Takes in a Player object and Game object and returns a placement
    in the form of a single piece with a proper flip, orientation, corners,
    and points. If no placement can be made function should return None.
    """
    possible_moves = player.possible_moves_opt(game)
    if len(possible_moves) > 0:
        return random.choice(possible_moves)
    return None

    # shape_options = [p for p in player.pieces]
    # while len(shape_options) > 0:
    #     piece_idx = random.randrange(len(shape_options))
    #     piece = shape_options[piece_idx]
    #     possibles = player.possible_moves([piece], game)
    #     # if there are not possible placements for that piece,
    #     # remove the piece from out list of pieces
    #     if possibles != []:
    #         return random.choice(possibles)
    #     else:
    #         shape_options.pop(piece_idx)
    # if the while loop finishes without returning a possible move,
    # there must be no possible moves left, return None
    # return None
