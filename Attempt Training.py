import gym
import numpy as np
import gym_anytrading
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

env = gym.make('forex-v0', window_size = 50, frame_bound = (50, 10000))

learning_rate = [1, 1e-3, 1e-2, 1e-1, 1e-4]
gammas = [0.99, 0.9, 0.8, 0.75, 0.5]
results = []

for learningr in learning_rate:
    for gammar in gammas:
        model = DQN(MlpPolicy, env, verbose=10, gamma = gammar, learning_rate=learningr)
        model.learn(total_timesteps=10000)

        observation = env.reset()

        while True:
            observation = observation[np.newaxis, ...]

            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)

            if done:
                results.append("\nGamma: " + str(gammar) + "\nLearning Rate:" + str(learningr) + "\nMax Possible Profit:" + str(env.max_possible_profit()) + "\nInfo:" + str(info) + "\n")
                break
    
    plt.cla()
    env.render_all()
    plt.show()

for result in results:
    print(result)