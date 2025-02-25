import os 
import numpy as np
#class to implement RARL training
class train_RARL ():

    #static wrapping function of training, it manages weights treatment
    def train(env, pro_policy_0, adv_policy_0, N_iter, N_pro, N_adv):

        env_name = env.unwrapped.get_id()
        path_pro, path_adv = "",""

        #inverted pendulum chioce
        if env_name== "ip":
            path_pro = "./pro_weights/pro_policy_weights_ip_RARL.zip"
            path_adv = "./adv_weights/adv_policy_weights_ip_RARL.zip"
            print("ip individuato")

        #walker2d choice
        elif env_name == "w2d":
            path_pro = "./pro_weights/pro_policy_weights_w2d_RARL3.zip"
            path_adv = "./adv_weights/adv_policy_weights_w2d_RARL3.zip"
            print("w2d individuato")

        #if weights are absent, train and save them   
        if not os.path.isfile(path_pro) or not os.path.isfile(path_adv):
            pro_policy, adv_policy, mean_reward_list = train_RARL.train_wrapped(env, pro_policy_0, adv_policy_0, N_iter, N_pro, N_adv )
            train_RARL.save_weights(pro_policy, adv_policy, env_name)
            np.save("last_dance.npy", mean_reward_list)

        else:
            print("Training already done")

    #static wrapping function of training, it manages the RARL training
    def train_wrapped(env, pro_policy_0, adv_policy_0, N_iter, N_pro, N_adv):

        #Initialize the policies
        pro_policy = pro_policy_0
        adv_policy = adv_policy_0
        mean_reward_list = []

        for i in range(N_iter):
            print(f"Iteration {i + 1}/{N_iter}")

            # Train the protagonist policy
            print("Training protagonist policy...")
            pro_policy = pro_policy.learn(total_timesteps=N_pro)

            done = False
            total_reward = 0
            obs, _ = env.reset()
            while not done:
                action, _ = pro_policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated 
                total_reward += reward
            print(f"Total reward: {total_reward}")
            mean_reward_list.append(total_reward)

            env.unwrapped.change_policy()

            # Train the adversary policy
            print("Training adversary policy...")
            adv_policy = adv_policy.learn(total_timesteps=N_adv)    
            env.unwrapped.change_policy()

        return pro_policy, adv_policy, mean_reward_list
            

    #saves the weights of the policies       
    def save_weights (pro_policy, adv_policy,env_name):

        if env_name =="ip":
            pro_policy.save("pro_policy_weights_ip")
            adv_policy.save("adv_policy_weights_ip")

        elif env_name == "w2d":
            pro_policy.save("pro_policy_weights_w2d")
            adv_policy.save("adv_policy_weights_w2d")


        
        
    

  