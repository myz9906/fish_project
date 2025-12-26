# Overview

This repository contains the code implementation for visual steering of fish locomotion using Reinforcement Learning (RL).

The core component is the interface script, `fish_deployment_interface.py`. This interface takes the current position of the **real fish** as input and outputs the target position for the **virtual fish** for the next time step.

This interface is integrated into the main control loop, `YizeMi_1VF_SysId_Ctr.py`. The control logic follows a two-stage process:

1. **Initialization:** The virtual fish guides the real fish in a **clockwise circular motion**.
2. **RL Guidance:** Once the system detects that the real fish has followed the virtual fish, it switches to the RL model to guide the fish along specific trajectories (e.g., letter shapes).



## Getting Started

### Installation

Follow these steps to set up the environment and run the code:

**Step 1: Clone the repository and enter the directory**

```shell
git clone https://github.com/myz9906/fish_project.git
cd fish_project
```

**Step 2: Install dependencies** We recommend using Conda to manage the environment.

```shell
conda env create -f environment.yml
```

**Step 3: Activate the environment**

```shell
conda activate fish_RL
```

### Updates in `YizeMi_1VF_SysId_Ctr.py`

#### 1. Circular Motion Direction

The initial circular motion has been updated from **counter-clockwise to clockwise**. This change ensures a smoother and more natural transition when the system switches from the circling phase to the specific letter-shaped trajectories.

#### 2. Model Selection for Debugging

We provide two pre-trained RL models with different parameters for debugging purposes. The primary difference between them is the **Reachable Set constraint** (the maximum allowable distance between the virtual and real fish) used during training:

- **Model A:** Constraint set to **0.01m**.
- **Model B:** Constraint set to **0.02m**.

You can switch between models by uncommenting the desired line in the code:

```python
# Use the 0.01m constraint model
MODEL_PATH_PROCESS = "checkpoints/letter_M_100_Hz_size_vr_ReachableSet_001/rl_model_9900000_steps.zip"

## Use the 0.02m constraint model
# MODEL_PATH_PROCESS = "checkpoints/letter_M_100_Hz_size_vr_ReachableSet_002/rl_model_9900000_steps.zip"
```

#### 3. Experimental Configurations

To evaluate the guidance performance under different conditions, the system is designed to cycle through different parameter sets during runtime. We currently test three specific configurations (rounds):

```python
Experiment_Configs = [
    {'rx': 0.01,  'ry': 0.01,  'type': 'box'},    # Config for Round 1
    {'rx': 0.02,  'ry': 0.02,  'type': 'box'},    # Config for Round 2
    {'rx': 0.01,  'ry': 0.01,  'type': 'circle'}, # Config for Round 3
]
```

**Logic:** When `self._Flag_round == Test_duration`, the system automatically switches to the next configuration in the list for the subsequent round of testing.
