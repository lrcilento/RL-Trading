import gym
import numpy as np
import gym_anytrading
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

env_maker = lambda: gym.make('forex-v0', window_size = 100, frame_bound = (100, 1000))

env = DummyVecEnv([env_maker])

policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[64, 64])])
model = A2C('MlpLstmPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=10000)

env = env_maker()
observation = env.reset()

while True:
    observation = observation[np.newaxis, ...]

    #action = env.action_space.sample()
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)

    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()