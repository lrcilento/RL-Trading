import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
from stable_baselines import DQN
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

env_maker = lambda: gym.make('forex-v0', df = FOREX_EURUSD_1H_ASK, frame_bound=(250,1500), window_size=250)
env = DummyVecEnv([env_maker])

model = DQN('MlpPolicy', env, prioritized_replay=True) 
model.learn(total_timesteps=1000)

env = gym.make('forex-v0', df = FOREX_EURUSD_1H_ASK, frame_bound=(250,1500), window_size=250)
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    #action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

