from gym.envs.registration import register

register(
    id='blokus-v0',
    entry_point='blokus.envs:BlokusEnv',
)

register(
    id='blokus-hard-v0',
    entry_point='blokus.envs:BlokusHardEnv',
)
