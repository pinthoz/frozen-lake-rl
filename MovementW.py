import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
DIAGONAL_LEFT_UP = 4
DIAGONAL_LEFT_DOWN = 5
DIAGONAL_RIGHT_UP = 6
DIAGONAL_RIGHT_DOWN = 7

class CustomActionWrapper(Wrapper):
    def __init__(self, env):
        super(CustomActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def step(self, action):
        # Ensure the action is within the valid range [0, 7]
        action = np.clip(action, 0, 7)

        if action == DIAGONAL_LEFT_UP:
            return self.step_diagonal(LEFT, UP)
        elif action == DIAGONAL_LEFT_DOWN:
            return self.step_diagonal(LEFT, DOWN)
        elif action == DIAGONAL_RIGHT_UP:
            return self.step_diagonal(RIGHT, UP)
        elif action == DIAGONAL_RIGHT_DOWN:
            return self.step_diagonal(RIGHT, DOWN)
        else:
            return self.env.step(action)

    def step_diagonal(self, action_horizontal, action_vertical):
        # Primeiro movimento vertical, depois horizontal
        original_state = self.env.s
        
        vertical_result = self.env.step(action_vertical) # down
        #print("vertical_result " + str(vertical_result))
        new_state_ver = vertical_result[0] 
        
        horizontal_result = self.env.step(action_horizontal) # left
        #print("horizontal_result " + str(horizontal_result))
        new_state = horizontal_result[0] 
        reward = horizontal_result[1]
        terminated = horizontal_result[2]
        truncated = horizontal_result[3]
        info = horizontal_result[4]
        
        # if the agent makes a diagonal move against a wall
        if new_state_ver == new_state or new_state_ver == original_state or new_state == original_state:
            reward -= 0.01
        #self.env.render()
        #print(f"New State: {new_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        return new_state, reward, terminated, truncated, info