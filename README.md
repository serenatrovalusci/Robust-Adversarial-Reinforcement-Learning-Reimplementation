Robust Adversarial Reinforcement Learning 

## Table of Contents
1. Extended_Inv_Pend.py : extended environment of InvertedPendulum-v5.
2. Extended_Walker2d.py : extended environment of Walker2d-v5.
7. train.py : file where we define the RARL algorithm.
8. main.py : training, evaluation and plotting.
9. adv_weights: a folder containing experimental weights. Default: adv_policy_weights_ip_RARL.zip for the Inverted Pendulum environment and  adv_policy_weights_w2d_RARL3.zip for the Walker2d environment.
10. pro_weights: a folder containing experimental weights. Default: pro_policy_weights_ip_RARL.zip for the Inverted Pendulum environment and  pro_policy_weights_w2d_RARL3.zip for the Walker2d environment.
11. ppo.py: ppo implementation from Stable-Baselines3.
12. provappo.py: training and evaluation of the ppo (stable-baselines3) algorithm.
13. "Trovalusci_Baldi.pdf": Presentation slides.
    
## Clone Repository:

git clone https://github.com/serenatrovalusci/Robust-Adversarial-Reinforcement-Learning-Reimplementation.git

## Run commands
As we alrready saved the weights, if you run the code, it will directly do the evaluation and rendering.

Run Inverted Pendulum simulation: 

"python main.py ip"

Run Walker 2d simulation:

"python main.py w2d"

## Reference paper: 

https://arxiv.org/abs/1703.02702

##Authors:

Serena Trovalusci
Andrea Baldi
