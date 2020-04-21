from blokus.envs.players.player import Player


class RandomPlayer(Player):
    def do_move(self):
        return self.sample_move()
