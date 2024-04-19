import torch
import torch.optim as optim
from Network import Network
from AgentTrainer import learning_rate, replay_buffer_size, minibatch_size, discount_factor, interpolation_parameter
from ReplayMemory import ReplayMemory
import random
import numpy as np
import torch.nn.functional as F


class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4 # increment but reset after 4 steps
        if self.t_step == 0: # learn every 4 steps
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon: float):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Unsqueeze -> Add extra dimension which is the batch. Which batch this state belongs to
        self.local_qnetwork.eval() # set in evaluation mode
        with torch.no_grad(): # make sure gradient calculations are disabled
            action_values = self.local_qnetwork(state)  # forward pass the state to the output layer

        # return to training mode
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, discount_factor):
        """ Update agents Q values based on sampled experiences """
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states, actions).detach().max(1)[0].unsqueeze(1) # 1 is the dimension of the actions, 0 is the max Q value, 1 is dimension of the batch
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones) # formula for the Q targets of the next states
        q_expected = self.local_qnetwork(states).gather(1, actions)
        #  compute loss (cost function) between the expected adn target Q calues
        loss = F.mse_loss(q_expected, q_targets) # mean square error loss
        #  backpropagate the loss to update the Q values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # perform a single optimization step
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_qnetwork, target_qnetwork, interpolation_parameter):
        """Soft update target model parameters using the local parameters."""
        for target_param, local_param in zip(target_qnetwork.parameters(), local_qnetwork.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

