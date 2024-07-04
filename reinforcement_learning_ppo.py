# Self-Driving Car (PPO)

# Import the libraries
import gym
import spinup.algos.pytorch.ppo.core as core

# Create the environment
env = gym.make('SelfDrivingCar-v0')

# Set the number of actions
num_actions = env.action_space.n

# Train the PPO model
core.ppo(env_fn=lambda: env, ac_kwargs=dict(hidden_sizes=[64, 64]))

# Test the PPO model
env.reset()
done = False

while not done:
  action, _states = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()

# Close the environment
env.close()
