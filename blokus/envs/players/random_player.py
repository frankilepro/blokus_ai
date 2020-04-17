import random


def Random_Player(player):
    """
    Takes in a Player object and Game object and returns a placement
    in the form of a single piece with a proper flip, orientation, corners,
    and points. If no placement can be made function should return None.
    """
    possible_moves = player.possible_moves_opt()
    if len(possible_moves) > 0:
        return random.choice(possible_moves)
    return None
