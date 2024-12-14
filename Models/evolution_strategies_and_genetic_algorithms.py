import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.multiprocessing import Pool
import os
import copy
import random

# Ensuring that Gymnasium's environments are registered for multiprocessing

gym.logger.set_level(40)  # Suppresses Gym warnings

# Building the Policy Network

class PolicyNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),  # Humanoid actions are in the range [-1, 1]
        )

    def forward(self, x):
        return self.network(x)

# Evaluating the policy

def evaluate_policy(policy_net, env_name='Humanoid-v4', seed=123, eval_episodes=3):
    env = gym.make(env_name)
    env.seed(seed)
    rewards = []
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state).detach().numpy()[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards)

# Altering the neural network weights and biases of the policy to introduce variability (Mutation)

def mutate_policy(policy_net, mutation_power=0.02):
    mutated_net = copy.deepcopy(policy_net)
    with torch.no_grad():
        for param in mutated_net.parameters():
            param += mutation_power * torch.randn_like(param)
    return mutated_net

# Combining parameters from two parent policies to create a child policy (Crossover)

def crossover_policy(policy_net1, policy_net2):
    child_net = copy.deepcopy(policy_net1)
    with torch.no_grad():
        for param1, param2, child_param in zip(policy_net1.parameters(), policy_net2.parameters(), child_net.parameters()):
            mask = torch.bernoulli(torch.full_like(param1, 0.5))
            child_param.copy_(mask * param1 + (1-mask) * param2)
    return child_net

# Implementing Parallel Evaluation to optimize the computational demands

def parallel_evaluate(nets, env_name):
    with Pool(mp.cpu_count()) as p:
        scores = p.starmap(evaluate_policy, [(net, env_name) for net in nets])
    return scores

# Implementing the Evolution Strategy Loop (Training)

def evolution_strategy(generations=10, population_size=50, top_k=10, env_name='Humanoid-v4'):
    input_size = gym.make(env_name).observation_space.shape[0]
    output_size = gym.make(env_name).action_space.shape[0]
    # Initializing the current population
    population = [PolicyNet(input_size, output_size) for _ in range(population_size)]
    for generation in range(generations):
        # Evaluating the current population
        scores = parallel_evaluate(population, env_name)
        # Selecting the top performers
        top_indices = np.argsort(scores)[-top_k:]
        top_nets = [population[i] for i in top_indices]
        print(f'Generation {generation}, Top Score: {max(scores)}')
        # Breeding the next population
        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(top_nets, 2)
            child_net = crossover_policy(parent1, parent2)
            child_net = mutate_policy(child_net)
            next_population.append(child_net)
        population = next_population

# Starting the training

if __name__ == '__main__':
    evolution_strategy()
