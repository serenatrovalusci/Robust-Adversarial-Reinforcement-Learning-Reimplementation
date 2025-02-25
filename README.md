Robust Adversarial Reinforcement Learning 

## Table of Contents
1. Extended_Inv_Pend.py : extended environment of InvertedPendulum-v5
2. Extended_Walker2d.py : extended environment of Walker2d-v5
3. adv_policy_weights_ip.py : weights for the trained adversarial agent InvertedPendulum N_iter= 100, N_adv=200
4. pro_policy_weights_ip.py : weights for the trained protagonist agent InvertedPendulum N_iter= 100, N_adv=200
5. pro_policy_weights_w2d.py : weights for the trained protagonist agent Walker2d N_iter= 500, N_adv=200
6. adv_policy_weights_w2d.py : weights for the trained adversarial agent Walker2d N_iter= 500, N_adv=200
7. train.py : file where we define the RARL algorithm
8. main.py : training and evaluation 


## Clone Repository:

git clone https://github.com/serenatrovalusci/Robust-Adversarial-Reinforcement-Learning-1.0.git

## Run commands
As we alrready saved the weights, if you run the files, it will directly do the evaluation and rendering.

Run Inverted Pendulumu: 

"python main.py ip"

Run Walker 2d:

"python main.py w2d"

## Compare with PPO Baselines 

Run file 'provappo.py' with N_iter*N_pro timesteps, using the desired gymnasum environment.

## Reference paper: 

https://arxiv.org/abs/1703.02702
