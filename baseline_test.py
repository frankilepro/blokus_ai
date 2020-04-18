import gym

from stable_baselines.common.env_checker import check_env
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, DQN

env = gym.make("blokus:blokus-v0")
# check_env(env)
# exit(1)

# Parallel environments
# env = make_vec_env("blokus:blokus-v0", n_envs=4)

print("starting training")
model = DQN(LnMlpPolicy, env, verbose=1)
# model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("blokus_weights")
print("finish training")
exit(1)

del model  # remove to demonstrate saving and loading

model = A2C.load("blokus_weights")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
