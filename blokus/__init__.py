from gym.envs.registration import register

register(
    id='blokus-hard-v0',
    entry_point='blokus.envs:BlokusEnv',
)

register(
    id='blokus-hard-greedy-v0',
    entry_point='blokus.envs:BlokusGreedyEnv',
)

register(
    id='blokus-simple-v0',
    entry_point='blokus.envs:BlokusSimpleEnv',
)

register(
    id='blokus-simple-greedy-v0',
    entry_point='blokus.envs:BlokusSimpleGreedyEnv',
)

register(
    id='blokus-duo-v0',
    entry_point='blokus.envs:BlokusDuoEnv',
)

register(
    id='blokus-duo-greedy-v0',
    entry_point='blokus.envs:BlokusDuoGreedyEnv',
)
