import gymnasium as gym
from stable_baselines3 import DQN
import os
from rewardwrappersb import CustomRewardWrapper
from sb3_contrib import TRPO






models_dir = "models/TRPO" 
logdir = "logs"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    

    
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env = CustomRewardWrapper(env)
env.reset()

model = TRPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 10000
iters = 0
for i in range(100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TRPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
#começar com i= 34 o ppo