import gymnasium as gym
import mujoco
from ppo import PPO
from train import train_RARL
import numpy as np

def evaluate_model(model, env, num_episodes):
    reward_per_episode = []  # List to store total rewards per episode
    print(f"Evaluating model on {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        max_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
            total_reward += reward
            max_reward += 1
        reward_per_episode.append(total_reward)
        print("Episode",episode + 1, ": Total Reward = ",total_reward, "Max Reward = ", max_reward)

    return reward_per_episode


import os 

class train_ppo ():

    def train(env, ppo_policy_0, N_iter, N_ppo):
        _, mean_reward_list = train_ppo.train_wrapped(env, ppo_policy_0, N_iter, N_ppo)
        np.save("mean_reward_list.npy", mean_reward_list)

        return mean_reward_list



    def train_wrapped(env, ppo_policy_0, N_iter, N_ppo):

        #Initialize the policies
        ppo_policy = ppo_policy_0
        mean_reward_list = []
        done = False
        total_reward = 0
        for i in range(N_iter):
            print(f"Iteration {i + 1}/{N_iter}")

            # Train the protagonist policy
            print("Training protagonist policy...")
            ppo_policy = ppo_policy.learn(total_timesteps=N_ppo)
            done = False
            total_reward = 0
            obs, _= env.reset()
            while not done:
                action, _ = ppo_policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated 
                total_reward += reward
            print(f"Total reward: {total_reward}")
            mean_reward_list.append(total_reward)



        return ppo_policy, mean_reward_list
            





env = gym.make("Walker2d-v5")
env_plot= gym.make("Walker2d-v5", render_mode = "human")

model = PPO("MlpPolicy", env, verbose=1)

train_ppo.train(env, model, 50, 1000)

reward_list_evaluation = evaluate_model(model, env, num_episodes=100)

np.save("reward_list_evaluation.npy", reward_list_evaluation)

env.close()
env_plot.close()
