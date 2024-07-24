import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromatch2024_experiments.q_networks import RNNQNetwork, LSTMQNetwork, GRUQNetwork
from neuromatch2024_experiments.agents import DQNAgent
import torch.nn as nn
from train import vanilla_train, mnist_train, cifar_train

configurations = [
    ("short-seq", "easy-seq", "RNN", "Rew-1"),
    ("short-seq", "easy-seq", "RNN", "Rew-10"),
    ("long-seq", "cifar10-seq", "LSTM", "Rew-10"),
    ("long-seq", "cifar10-seq", "LSTM", "Rew-100")
]

networks = {
    "RNN": RNNQNetwork,
    "LSTM": LSTMQNetwork,
    # "GRU": GRUQNetwork
}

state_sizes = {
    "short-seq": 6,
    "long-seq": 11
}

experiments = {}

for item in configurations:
    experiments[item] = {}
    if item[1] == 'easy-seq':
        state_size = state_sizes[item[0]]
        action_size = state_size
    elif item[1] == 'cifar10-seq':
        # Logits for MNIST and CIFAR10 are 10-dimensional
        state_size = 10
        if item[0] == 'short-seq':
            action_size = 6
        elif item[0] == 'long-seq':
            action_size = 11
    # if item[0] == "easy-seq":
        # state_size = 6
        # action_size = state_size
    # elif item[0] == 'cifar10-seq':
        # Logits for MNIST and CIFAR10 are 10-dimensional
        # state_size = 10
        # if seq_type == 'short-seq':
            # action_size = 6
        # elif seq_type == 'long-seq':
            # action_size = 11
    # if item[0] == "short-seq":
        # experiments[item]['state_size'] = 6
    # elif item[0] == "long-seq":
        # experiments[item]['state_size'] = 11
    if item[1] == "easy-seq":
        experiments[item]['train_function'] = vanilla_train
    else:
        experiments[item]['train_function'] = cifar_train
    if item[2] == "RNN":
        experiments[item]['q_network'] = RNNQNetwork
    elif item[2] == "LSTM":
        experiments[item]['q_network'] = LSTMQNetwork
    if item[3] == "Rew-1":
        experiments[item]['reward'] = 1
        rew = 1
    elif item[3] == "Rew-10":
        experiments[item]['reward'] = 10
        rew = 10
    elif item[3] == "Rew-100":
        experiments[item]['reward'] = 100
        rew = 100
    hidden_size = 64 if item[0] == "short-seq" else 128

    model_path = f"models/{item[0]}_{item[1]}_MSELoss_{item[2]}_rew-{rew}.pth"
    learning_curve = f"models/{item[0]}_{item[1]}_MSELoss_{item[2]}_rew-{rew}.npy"

    experiments[item]['hidden_size'] = hidden_size

    experiments[item]['loss'] = nn.MSELoss
    experiments[item]['action_size'] = action_size
    experiments[item]['state_size'] = state_size
    experiments[item]['capacity'] = 100000
    experiments[item]['batch_size'] = 32
    experiments[item]['lr'] = 0.001 if item[0] == "short-seq" else 0.0001
    experiments[item]['gamma'] = 0.99
    experiments[item]['model_path'] = model_path
    experiments[item]['mode'] = 'train-from-zero'
    experiments[item]['train_steps'] = 10000
    experiments[item]['learning_curve_path'] = learning_curve

    # if item == ("long-seq", "cifar10-seq", "LSTM", "Rew-100"):
    # experiments[item]['hidden_size'] = 512
