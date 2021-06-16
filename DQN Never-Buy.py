import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from gym_anytrading.datasets import STOCKS_GOOGL
from stable_baselines import DQN
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

env_maker = lambda: gym.make('stocks-v0', df = STOCKS_GOOGL, frame_bound=(250,1500), window_size=250)
env = DummyVecEnv([env_maker])

model = DQN('MlpPolicy', env, verbose=1) 
model.learn(total_timesteps=100000)

env = gym.make('stocks-v0', df = STOCKS_GOOGL, frame_bound=(250,1500), window_size=250)
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

