import gymnasium as gym
from gymnasium import RewardWrapper

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        # Define your custom rewards for each square
        self.custom_rewards = {
            b'S': 0.0,  # Start
            b'F': -0.01,  # Frozen
            b'H': -1,  # Hole
            b'G': 1    # Goal
        }

    def step(self, action):
        # Take a step in the environment
        step_result = self.env.step(action)
        #print("action " + str(action))
        # Unpack the step result with the additional 'prob' dictionary
        new_state, reward, terminated, truncated, info = step_result
        
        # Customize the reward based on the current state
        current_state = self.env.unwrapped.s
        row = current_state // self.ncol
        col = current_state - row * self.ncol
        tile_type = self.desc[row, col]
        
        # the possible "reward" values ( that come from the movement wrapper ) are:
        # -0.01 ("incorrect" diagonal move), 1 (goal), and 0 (everything else)
        new_reward = reward + self.custom_rewards.get(tile_type, reward)
        # new reward is the reward  from the movement wrapper + the reward from the custom rewards dictionary
        # 0 + -0.01 = 0.01 for frozen
        # -0.01 + -0.01 = -0.02 for incorrect diagonal move (against a wall -> counts as 2 steps)
        # 0 + 0 = 0 for start
        # 1 + 1 = 2 for goal
        # 0 + -1 = -1 for hole
        
        #print(f"Current state: {tile_type} {new_state}, new reward: {new_reward}")
        
        return new_state, new_reward, terminated, truncated, info