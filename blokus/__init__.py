from gym.envs.registration import register

register(
    id='blokus-hard-v0',
    entry_point='blokus.envs:BlokusEnv',
)

register(
    id='blokus-simple-v0',
    entry_point='blokus.envs:BlokusSimpleEnv',
)

register(
    id='blokus-duo-v0',
    entry_point='blokus.envs:BlokusDuoEnv',
)
