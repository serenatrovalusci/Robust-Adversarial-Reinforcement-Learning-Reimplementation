# Robust Adversarial Reinforcement Learning (RARL)

A reimplementation of [*Robust Adversarial Reinforcement Learning*](https://arxiv.org/abs/1703.02702) (Pinto et al., 2017), tested on two MuJoCo continuous control environments: **Inverted Pendulum** and **Walker2d**.

> **Authors:** Serena Trovalusci · Andrea Baldi  
> **Slides:** [`Trovalusci_Baldi.pdf`](Trovalusci_Baldi.pdf)

---

## What is RARL?

Standard RL agents can be brittle — small perturbations to the environment (wind, friction, sensor noise) can cause them to fail. RARL addresses this by framing training as a **two-player zero-sum game**:

-  **Protagonist** — learns to complete the task.
-  **Adversary** — applies destabilising forces to the protagonist, trying to make it fail.

The two agents are trained in alternation using **Proximal Policy Optimization (PPO)**. The result is a protagonist robust to a wide range of adversarial perturbations at test time.

```
┌─────────────────────────────────────────────┐
│              RARL Training Loop             │
│                                             │
│  for each iteration:                        │
│    1. Fix adversary  → train protagonist    │
│    2. Fix protagonist → train adversary     │
└─────────────────────────────────────────────┘
```

---

## Environments

| Environment | Base | Description |
|---|---|---|
| **Inverted Pendulum** | `InvertedPendulum-v5` | Balance a pole on a cart; adversary applies lateral forces |
| **Walker2d** | `Walker2d-v5` | Bipedal locomotion; adversary applies torques to the joints |

Both environments are extended versions that expose an adversarial force interface on top of the standard MuJoCo Gymnasium API.

---

## Repository Structure

```
.
├── Extended_Inv_Pend.py        # Custom InvertedPendulum-v5 with adversarial forces
├── Extended_Walker2d.py        # Custom Walker2d-v5 with adversarial forces
├── train.py                    # RARL training algorithm
├── main.py                     # Entry point: training, evaluation, and plotting
├── ppo.py                      # PPO implementation (Stable-Baselines3)
├── provappo.py                 # Standalone PPO baseline training & evaluation
├── adv_weights/                # Pre-trained adversary policy weights
│   ├── adv_policy_weights_ip_RARL.zip
│   └── adv_policy_weights_w2d_RARL3.zip
├── pro_weights/                # Pre-trained protagonist policy weights
│   ├── pro_policy_weights_ip_RARL.zip
│   └── pro_policy_weights_w2d_RARL3.zip
└── Trovalusci_Baldi.pdf        # Presentation slides
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/serenatrovalusci/Robust-Adversarial-Reinforcement-Learning-Reimplementation.git
cd Robust-Adversarial-Reinforcement-Learning-Reimplementation
```

### 2. Install dependencies

```bash
pip install gymnasium[mujoco] stable-baselines3 torch numpy matplotlib
```

> Requires Python ≥ 3.8 and a working MuJoCo installation.

### 3. Run the simulation

Pre-trained weights are included, so running the commands below will **skip training and go straight to evaluation and rendering**.

```bash
# Inverted Pendulum
python main.py ip

# Walker2d
python main.py w2d
```

To retrain from scratch, remove or rename the contents of `adv_weights/` and `pro_weights/` before running.

---

## Results

Agents trained with RARL demonstrate significantly more stable behaviour under adversarial perturbations compared to a vanilla PPO baseline, consistent with the findings of the original paper.

### Inverted Pendulum

The Inverted Pendulum uses the **standard RARL algorithm from the paper** with no custom reward shaping. During training, RARL and PPO achieve comparable cumulative reward (PPO converges slightly faster since it faces no adversarial pressure). During evaluation, both policies reach maximum reward — confirming the protagonist successfully learns to balance under disturbance even when trained against an adversary.

### Walker2d

Walker2d shows the clearest benefit of RARL: the protagonist trained adversarially **outperforms the PPO baseline at evaluation time**, despite lower training rewards. Three reward variants were explored:

| Config | Iterations | N_pro = N_adv | Reward shaping | Result |
|---|---|---|---|---|
| RARL 1 | 500 | 100 | Default Walker2d reward | Basic forward locomotion |
| RARL 2 | 500 | 100 | Velocity, posture & foot angle penalties | Higher control, lower raw reward |
| **RARL 3** *(best)* | **200** | **150** | **Sinusoidal gait + periodicity rewards** | **Most natural and stable gait** |

#### Best configuration: Sinusoidal Reward Shaping (RARL 3)

The key insight behind RARL 3 is that **enforcing periodicity in leg motion leads to both greater stability and a more natural gait**. The custom reward function encourages the agent to move its legs in a coordinated, sinusoidal pattern rather than adopting degenerate locomotion strategies. Specifically, the reward was shaped by:

- Rewarding a **large angular difference between the two thighs** (wide, alternating steps)
- Rewarding **opposite angular velocity signs** of the thighs (true alternating motion)
- Rewarding **knee angular speeds** (active, dynamic leg bending)
- Encouraging **sinusoidal joint trajectories** over time
- Penalising **jerk** (excessively fast or jerky movements)

This combination promotes a smooth, periodic, human-like walking cycle — and when combined with adversarial training, produces a protagonist that is both graceful and robust to external disturbances.

---

## Demo — Best Performance

**Inverted Pendulum** *(standard RARL)*

https://github.com/user-attachments/assets/38420451-5d72-4a85-91de-740a09f1db13

**Walker2d — RARL 3** *(sinusoidal reward shaping)*

https://github.com/user-attachments/assets/0229c6a6-f8df-4c38-837a-82af53f1f0fe

## Reference

```bibtex
@article{pinto2017robust,
  title   = {Robust Adversarial Reinforcement Learning},
  author  = {Pinto, Lerrel and Davidson, James and Sugumar, Rahul and Gupta, Abhinav},
  journal = {arXiv preprint arXiv:1703.02702},
  year    = {2017}
}
```
