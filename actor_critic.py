import argparse
import sys
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from actor_network import ActorNetwork
from critic_network import CriticNetwork

VALID_ENVS = ['acrobot', 'mountaincar']

def run_actor_critic(env, num_episodes=1000, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, timeout=None):
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    actor = ActorNetwork(state_space, action_space)
    critic = CriticNetwork(state_space)
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        I = 1.0 
        is_done = False    
        epi_reward = 0

        while not (is_done):
            curr_state = torch.from_numpy(state).float().unsqueeze(0)
            action_logits = actor.forward(curr_state)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            is_done = terminated or truncated
            epi_reward += reward
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float)
            value_curr = critic.forward(curr_state)

            with torch.no_grad():
                if is_done:
                    value_next = torch.tensor([[0.0]])
                else:
                    value_next = critic.forward(next_state_tensor)

            target = reward_tensor + gamma * value_next
            delta = target - value_curr

            critic_loss = F.mse_loss(value_curr, target)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            log_prob = dist.log_prob(action)
            actor_loss = - (I * delta.detach() * log_prob)  
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            I *= gamma
            state = next_state
        rewards.append(epi_reward)
        if (episode + 1) % 50 == 0:
            print(f"Episode - {episode+1}/{num_episodes} | Generated Reward: {epi_reward}")

    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a one-step Actor-Critic agent on the chosen environment - MountainCar or Acrobot")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name - 'Acrobot', 'MountainCar'"
    )
    args = parser.parse_args()
    env = args.env.strip().lower()
    if env not in VALID_ENVS:
        print(f'Environment not valid')
        print(f'Choose from - {VALID_ENVS}')
        sys.exit(1)
    
    if env == 'acrobot':
        gym_env = gym.make('Acrobot-v1')
        rewards = run_actor_critic(gym_env, num_episodes=500, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99)
    elif env == 'mountaincar':
        gym_env = gym.make('MountainCar-v0')
        rewards = run_actor_critic(gym_env, num_episodes=1500, actor_lr=1e-4, critic_lr=1e-3, gamma=0.97, timeout=1000)
    
