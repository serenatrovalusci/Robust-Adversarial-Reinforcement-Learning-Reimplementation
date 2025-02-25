import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Adversarial_InvertedPendulum(gym.Env):


    def __init__(self, pro_policy=None, adv_policy=None): 
        super().__init__()
        # Load the base environment
        self.env = gym.make("InvertedPendulum-v5") 
        self.env_id = 'ip'
        self.observation_space = self.env.observation_space #observation space
        self.action_space = spaces.Box(low=-3, high=3, dtype=np.float32) #action space
        self.locked_policy = True # By default, the adversary policy is locked and cannot be changed
        self.pro_policy = pro_policy #protagonist policies
        self.adv_policy = adv_policy #adversarial policies
        self.last_obs = None  #useful for training
    
    def step(self, action): 
        pro_action, adv_action = 0, 0 
        #this locks not training policy and asks it only to predict, it helps dealing with stable-baseline PPO module 
        if self.locked_policy:
            adv_action,_ = self.adv_policy.predict(self.last_obs, deterministic=True)
            pro_action = action
        else: 
            pro_action,_ = self.pro_policy.predict(self.last_obs, deterministic=True)
            adv_action = action

        #clip on adversary action
        adv_action = adv_action/3.0 # Scale the adversary action to be between -1 and 1
       
        #summing actions
        action_sum = pro_action + adv_action
        #calling step
        obs, protagonist_reward, terminated, truncated, info = self.env.step(action_sum)
         # Adversary's reward is the negative of protagonist's reward
        adversary_reward = -protagonist_reward

        #torque computing
        torque = adv_action  # Torque applied by the adversary
        # Apply adversary action
        # Pendulum parameters
        mass = 1.0  # kg
        length = 1.0  # m
        inertia = (1/3) * mass * (length ** 2)  # Moment of inertia for a rod pivoted at one end
        dt = 0.02  # Time step
        theta = obs[1]
        theta_dot = obs[3] 
        theta_ddot = torque / inertia  # Angular acceleration
        theta_dot += theta_ddot * dt  # Update angular velocity
        theta += theta_dot * dt  # Update angle
        obs[1] = theta
        obs[3] = theta_dot

        #update last observation    
        self.last_obs = obs

        #returns right values on which policy's training
        if self.locked_policy:
            return obs, protagonist_reward, terminated, truncated, info
        else:
            return obs, adversary_reward, terminated, truncated , info

    #changes locked policy
    def change_policy(self):
        self.locked_policy = not self.locked_policy
        return

    def set_policies(self, pro_policy, adv_policy):
        self.pro_policy = pro_policy 
        self.adv_policy = adv_policy
        return 
    
    def reset(self, *, seed=None, options=None):
        self.last_obs = None  # Reset last_obs
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs,info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
    
    def get_id(self):
        return self.env_id  # Method to retrieve the ID
    