from gym.envs.registration import register

register(
    id='blokus-hard-v0',
    entry_point='blokus_gym.envs:BlokusEnv',
)

register(
    id='blokus-hard-greedy-v0',
    entry_point='blokus_gym.envs:BlokusGreedyEnv',
)

register(
    id='blokus-simple-v0',
    entry_point='blokus_gym.envs:BlokusSimpleEnv',
)

register(
    id='blokus-simple-greedy-v0',
    entry_point='blokus_gym.envs:BlokusSimpleGreedyEnv',
)

register(
    id='blokus-duo-v0',
    entry_point='blokus_gym.envs:BlokusDuoEnv',
)

register(
    id='blokus-duo-greedy-v0',
    entry_point='blokus_gym.envs:BlokusDuoGreedyEnv',
)
