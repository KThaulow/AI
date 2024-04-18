import gymnasium as gym

# Part 2  - Training the AI

env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape  # vector of 8 inputs
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('state shape: ', state_shape)
print('State size: ', state_size)
print('Number actions: ', number_actions)

# Initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99  # gamma
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3



