from blokus_gym.envs.players.player import Player


class AiPlayer(Player):
    next_move = None  # Needs to be set by the exterior

    def do_move(self):
        return self.next_move
