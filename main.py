import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Extended_Inv_Pend import Adversarial_InvertedPendulum
from gymnasium.envs.registration import register
from train import train_RARL
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# it initializes pro and adv PPOs
def initialize_ppo(env,learning_rate=3e-4):

    env_name = env.unwrapped.get_id()

    #it check if weights are already present
    if env_name== "ip":
        path_pro = "./pro_weights/pro_policy_weights_ip_RARL.zip"
        path_adv = "./adv_weights/adv_policy_weights_ip_RARL.zip"
        print("ip individuato")

     #walker2d choice
    elif env_name == "w2d":
        path_pro = "./pro_weights/pro_policy_weights_w2d_RARL3.zip"
        path_adv = "./adv_weights/adv_policy_weights_w2d_RARL3.zip"
        print("w2d individuato")

    if os.path.isfile(path_pro) and os.path.isfile(path_adv):
        print(f"Loading existing PPO models for {env_name}...")
        pro_ppo = PPO.load(path_pro, env=env)
        adv_ppo = PPO.load(path_adv, env=env)
    else: #if not it creates two new PPOs
        print(f"No pre-trained model found for {env_name}, initializing new PPO models...")
        pro_ppo = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=1, device='cpu')
        adv_ppo = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=1, device='cpu')

    return pro_ppo, adv_ppo


#it evaluates the model
def evaluate_model(model, env, num_episodes=5):
    print(f"Evaluating model on {num_episodes} episodes...")
    reward_per_episode = []
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
        print("Episode ",episode + 1, ": Total Reward = ",total_reward, " Timesteps = ", max_reward)

    return reward_per_episode  

def plot_cumulative_reward  (mean_reward_list_rarl, mean_reward_list_ppo):
  
    cumulative_rewards_rarl = np.cumsum(mean_reward_list_rarl)  # Compute cumulative sum
    cumulative_rewards_ppo = np.cumsum(mean_reward_list_ppo)  # Compute cumulative sum
    episodes = np.arange(1, len(mean_reward_list_rarl) + 1)  # Episode numbers

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_rewards_rarl, label="RARL", color="red", marker="o")
    plt.plot(episodes, cumulative_rewards_ppo, label="PPO", color="blue", marker="o")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Evaluation Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(arg):
    #it begins from initializing training and evaluation environments either for inv pend or w2d
    if arg == "ip":
        register(
        id='InvertedPendulumExtended-v1',  # environment ID
        entry_point='Extended_Inv_Pend:Adversarial_InvertedPendulum', 
        ) 
        ext_env = gym.make('InvertedPendulumExtended-v1')
        env = gym.make('InvertedPendulum-v5', render_mode="human")
        env_plot = gym.make('InvertedPendulum-v5', render_mode=None)

    elif arg == "w2d":
        register(
        id='Walker2dExtended-v1',  #environment ID
        entry_point='Extended_Walker2d:Adversarial_Walker2d',
        ) 
        ext_env = gym.make('Walker2dExtended-v1')
        env = gym.make('Walker2d-v5', render_mode="human")
        env_plot = gym.make('Walker2d-v5', render_mode=None)

    # PPO initialization 
    pro_ppo, adv_ppo = initialize_ppo(ext_env) 
    ext_env.unwrapped.set_policies(pro_ppo, adv_ppo) #setting policies to environemnt


    # it trains with the training environment
    #mean_reward_list_ppo = np.load("mean_reward_list.npy", allow_pickle=True)
    train_RARL.train(ext_env,pro_ppo,adv_ppo, N_iter = 200, N_pro=1500, N_adv= 1500)

    # it evaluates obtained model through "classical" environment
    reward_list_eval_rarl = evaluate_model(pro_ppo, env, num_episodes=5)
    #reward_list_eval_ppo= np.load("reward_list_evaluation.npy", allow_pickle=True)

    #plots the performances
    #plot_cumulative_reward(mean_reward_list_RARL, mean_reward_list_ppo)
    #plot_cumulative_reward(reward_list_eval_rarl, reward_list_eval_ppo)

    # it closes the environment
    ext_env.close()
    env.close()
    env_plot.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scelta dell'environment")
    parser.add_argument("arg", type=str)
    args = parser.parse_args()
    main(args.arg)