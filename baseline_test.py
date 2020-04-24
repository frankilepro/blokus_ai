import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, DQN
from blokus_gym.envs.blokus_simple_env import BlokusSimepleEnv

env = gym.make("blokus_gym:blokus-simple-v0")
check_env(env)
exit(1)

# Parallel environments
# env = make_vec_env(BlokusEnv(), n_envs=4)
env = make_vec_env("blokus_gym:blokus-simple-v0", n_envs=4)

print("starting training")
# model = DQN(LnMlpPolicy, env, verbose=1)
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="logdir")
model.learn(total_timesteps=25000)
model.save("blokus_weights")
print("finish training")
input()

del model  # remove to demonstrate saving and loading

model = A2C.load("blokus_weights")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
