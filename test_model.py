import gymnasium as gym
from stable_baselines3 import DQN
from sb3_contrib import TRPO
from rewardwrapper import CustomRewardWrapper
import numpy as np

models_dir = "models/TRPO"

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
env = CustomRewardWrapper(env)
env.reset()
model_path = f"{models_dir}/990000"
model = TRPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs,info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        print("action " + str(action))
        action = np.argmax(action)
        new_state, rewards, done, truncated, info = env.step(action)
        env.render()
        print(rewards)