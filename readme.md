
# Actor-Critic (one-step)
This folder contains a one-step Actor-Critic implementation using PyTorch and Gymnasium. The main script is `actor_critic.py` which trains an agent on either the Acrobot or MountainCar environments.

**Overview:**
- `actor_critic.py`: training script (CLI) that accepts an `--env` argument.
- `actor_network.py`: actor network definition.
- `critic_network.py`: critic network definition.

**Requirements**
- Python 3.8 or newer
- PyTorch
- Gymnasium
- NumPy

Train on Acrobot:
```bash
python3 actor_critic.py --env acrobot
```

Train on MountainCar:
```bash
python3 actor_critic.py --env mountaincar
```

The script will run for a default number of episodes (500 for acrobot and 1500 for mountain-car) and will print periodic progress (reward summaries every 50 episodes).

**Arguments**
- `--env`: (required) environment name. Valid values: `acrobot`, `mountaincar`.

