import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Setting the hyperparameters
state_dim = 96 * 96 * 3  # Use this placeholder until env is defined
action_dim = 3  # Use this placeholder until env is defined
hidden_dim = 256
batch_size = 128
learning_rate = 1e-4
gamma = 0.99
tau = 0.005
buffer_size = int(1e5)
min_buffer_size = 1000
num_episodes = 500
max_timesteps = 1000

# Building the Actor Network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Tanh for bounded actions
        return action

# Building the Critic Network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Implementing the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# Implementing the DDPG Algorithm
class DDPG:
    def __init__(self):
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def update(self, batch_size):
        if len(self.replay_buffer) < min_buffer_size:
            return  # Exit if not enough samples
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        # Update of the Critic network
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update of the Actor network
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update of the Critic Target and the Actor Target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    @staticmethod
    def soft_update(local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Implementing the Training Loop
def train(env, agent, num_episodes, max_timesteps):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32)  # Flatten state
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.actor(state).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.update(batch_size)

            if done:
                break

        print(f'Episode {episode}: Reward = {episode_reward}')
    env.close()

# Setting up the environment
env = gym.make('CarRacing-v3')
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.shape[0]

# Creating the agent
ddpg_agent = DDPG()

# Starting the training
train(env, ddpg_agent, num_episodes, max_timesteps)
