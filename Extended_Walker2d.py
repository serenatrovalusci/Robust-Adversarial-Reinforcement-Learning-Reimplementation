import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Adversarial_Walker2d(gym.Env):   # Adversarial_InvertedPendulum is a class that extends the InvertedPendulum-v5 environment to include an adversary that can apply forces to the cart.

    def __init__(self, pro_policy=None, adv_policy=None): 
        super().__init__()
        # Load the base environment
        self.env = gym.make("Walker2d-v5")
        self.env_id = 'w2d'
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32) #defines action space
        self.locked_policy = True # By default, the adversary policy is locked and cannot be changed
        self.pro_policy = pro_policy  #includes the two policies
        self.adv_policy = adv_policy
        self.ts_count = 0 #helps with sinuisoidal function
        self.last_obs = None #helps with reward reshape
        self.last_act = None 
    
    def step(self, action): 

        pro_action, adv_action = 0, 0 
        #choses which policy to train
        if self.locked_policy:
            adv_action,_ = self.adv_policy.predict(self.last_obs, deterministic=True)
            pro_action = action
        else: 
            pro_action,_ = self.pro_policy.predict(self.last_obs, deterministic=True)
            adv_action = action

        #scales and clips actions
        adv_action = adv_action/5.0 # Scale the adversary action to be between -0.5 and 0.5
        action_sum = np.clip(pro_action + adv_action, -1, 1) #scale action sum in order to balance joint control. PROVA: è aumentata la reward

        #executes step
        obs, protagonist_reward, terminated, truncated, info = self.env.step(action_sum)
        
        
        self.last_obs = obs
        protagonist_reward = 0 


        #encorage to keep ideal speed
        x_velocity = info["x_velocity"]
        #max_speed = 1.5  # max speed
        #min_speed = 0.5 # min speed
        ideal_speed = 1.5
        #beta_rew = 1.5  # reward for desired speed
        beta_pen = 0.4  # penalty for difference from ideal speed

        # if x_velocity < max_speed and x_velocity > min_speed:
        #     protagonist_reward += beta_rew * (abs(x_velocity - ideal_speed) / (max_speed - min_speed))
        # else:
        #     protagonist_reward -= beta_pen 

        protagonist_reward -= beta_pen * abs(x_velocity - ideal_speed) #ideal speed setting
        



        #penalizes when the walker jumps too high rewards when stays in below threshold
        z_jump_threshold = 0.2 # soglia oltre la quale il walker viene penalizzato
        alpha_pen = 1.0
        alpha_rew = 0.1
        current_z = info["z_distance_from_origin"]
        if abs(current_z) > z_jump_threshold:
            protagonist_reward += -alpha_pen * abs(current_z)
        else:
            protagonist_reward += alpha_rew 
        
   
        
        
        #reward when the foot angle is near 0   
        right_foot_angle = obs[4]
        left_foot_angle = obs[7]
        foot_angle_threshold = 0.05
        gamma = 0.2
        if abs(right_foot_angle) > foot_angle_threshold:
            protagonist_reward -= gamma * (abs(right_foot_angle))
        if abs(left_foot_angle) > foot_angle_threshold:
            protagonist_reward -= gamma * (abs(left_foot_angle))




        # sinuisoidal reward for hips angle

        rthigh_ang_vel = obs[11]
        lthigh_ang_vel = obs[14]
        rknee_ang_vel = obs[12]
        lknee_ang_vel = obs[15]
        rthigh_ang = obs[2]
        lthigh_ang = obs[5]
        #rknee_ang = obs[3]
        #lknee_ang = obs[6]
        #omega = 0.6
        #amplitude = 0.40
        #offset = 0.17
        #amplitude_hip = 0.35
        # amplitude_knee = 0.4 
        #phase_hip = np.pi 
        # phase_rknee = np.pi/2
        # phase_lknee = 3 * np.pi / 2
        #time = self.ts_count * 0.002
        #theta = 0.1
        #theta_amp = 0.5

        #reward for walk movements
        #if abs((rthigh_ang-lthigh_ang) - (amplitude * np.sin(2 * np.pi * omega * time))) < 0.05:
        #   protagonist_reward += theta_amp
        #protagonist_reward -= theta * abs((rthigh_ang-lthigh_ang) - (amplitude * np.sin(2 * np.pi * omega * time)))
        #if abs((rthigh_ang_vel-lthigh_ang_vel) - (amplitude * 2 * np.pi * omega * np.cos(2 * np.pi * omega * time))) < 0.05:
            #protagonist_reward += theta
        #protagonist_reward -= theta * abs((rthigh_ang_vel-lthigh_ang_vel) - (amplitude * 2 * np.pi * omega * np.cos(2 * np.pi * omega * time)))
        
        
        # #reward for right hip
        #if abs(rthigh_ang - (offset + amplitude_hip * np.sin(2 * np.pi * omega * time))) < 0.05:
        #    protagonist_reward += theta
        #protagonist_reward -= theta * abs(rthigh_ang - (offset + amplitude_hip*np.sin(2 * np.pi * omega * time)))
        #protagonist_reward -= theta * abs(rthigh_ang_vel - amplitude_hip*2*np.pi*omega*np.cos(2 * np.pi * omega * time))
        
        # #reward for left hip
        #if abs(lthigh_ang - (offset + amplitude_hip * np.sin(2 * np.pi * omega * time + phase_hip))) < 0.05:
        #    protagonist_reward += theta
        #protagonist_reward -= theta * abs(lthigh_ang - (offset + amplitude_hip*np.sin(2 * np.pi * omega * time + phase_hip)))
        #protagonist_reward -= theta * abs(lthigh_ang_vel - amplitude_hip*2*np.pi*omega*np.cos(2 * np.pi * omega * time + phase_hip))

        #control penality


        #rewards difference in thigh angular position
        open_gamma = 0.2
        if abs(rthigh_ang - lthigh_ang) > 0.15 :
            protagonist_reward += open_gamma

        #rewards opposite thigh angular speeds
        opsp_gamma = 0.15
        if np.sign(rthigh_ang_vel) != np.sign(lthigh_ang_vel):
            protagonist_reward += opsp_gamma

        #improves smoothness of movements by penalizing too high variations    
        phi = 0.05
        if self.ts_count > 0:
            delta_action = action - self.last_act
            delta_u = np.linalg.norm(delta_action)
            protagonist_reward -= phi * delta_u

        #rewards movements of knees
        psi = 0.03
        protagonist_reward += psi * abs(rknee_ang_vel)
        protagonist_reward += psi * abs(lknee_ang_vel)


        #protagonist_reward += psi * abs(rthigh_ang_vel)
        #protagonist_reward += psi * abs(lthigh_ang_vel)
        #if abs(rthigh_ang - lthigh_ang) > 0.25:
        #    protagonist_reward += phi
        #if abs(rthigh_ang_vel) > 0.3 or (lthigh_ang_vel) > 0.3:
        #    protagonist_reward += phi
        #if abs(rknee_ang_vel) > 0.3 or abs(lknee_ang_vel) > 0.3:
        #    protagonist_reward += phi        


        #penalty if the torso is not vertical        
        torso_angle = obs[1]
        #torso_threshold = 0.05
        torso_desired = 0.17
        #delta  = 0.2
        #delta_starving = 0.1
        delta_control = 0.8
        # if torso_angle > torso_threshold:
        #     protagonist_reward -= delta
        # if torso_angle <= 0:
        #     protagonist_reward -= delta_starving  
        protagonist_reward -= delta_control * abs(torso_angle - torso_desired)


        #updates current timesteps
        self.ts_count += 1
        
        #survival reward
        psi = 1.0
        protagonist_reward += psi   
        
        

        adversary_reward = -protagonist_reward # Adversary's reward is the negative of protagonist's reward
        
        self.last_act = action
        #choose whether to return protagonist's or adversary's reward
        if self.locked_policy:
            return obs, protagonist_reward, terminated, truncated, info
        else:
            return obs, adversary_reward, terminated, truncated , info



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
        #self.jerk = [[] for _ in range(4)]
        #self.jerk.append([0,0,0,0])
        self.ts_count = 0
        self.last_act = None
        return obs,info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
    
    def get_id (self):
        return self.env_id
    

