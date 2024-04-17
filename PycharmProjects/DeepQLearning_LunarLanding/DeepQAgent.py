import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple


# Part 1 - Building the AI
# Creating the architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, state_size, action_size, seed=42) -> None:
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)  # First fully connected layer, in_features is the input layer
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, action_size)  # Third fully connected layer, out_features is the output layer

    def forward(self, state):
        """
        Forward pass of the network
        """
        x = self.fc1(state)  # Propagate the signal from input layer to the first fully connected layer
        x = F.relu(x)  # rectifier activation function
        x = self.fc2(x)  # Propagate the signal from first fully connected layer to the second fully connected layer
        x = F.relu(x)
        return self.fc3(x)

#Part 2  - Training the AI
