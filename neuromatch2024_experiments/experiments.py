from q_networks import RNNQNetwork, LSTMQNetwork, GRUQNetwork
from agents import DQNAgent
import torch.nn as nn
from train import vanilla_train, mnist_train, cifar_train

# Define configurations and parameters
configurations = [
    ("short-seq", "easy-seq"),
    # ("short-seq", "mnist-seq"),
    ("short-seq", "cifar10-seq"),
    ("long-seq", "easy-seq"),
    # ("long-seq", "mnist-seq"),
    ("long-seq", "cifar10-seq")
]

networks = {
    "RNN": RNNQNetwork,
    "LSTM": LSTMQNetwork,
    # "GRU": GRUQNetwork
}

losses = {
    "MSELoss": nn.MSELoss,
    # "SmoothL1Loss": nn.SmoothL1Loss
}

state_sizes = {
    "short-seq": 6,
    "long-seq": 11
}

train_functions = {
    "easy-seq": vanilla_train,
    # "mnist-seq": mnist_train,
    "cifar10-seq": cifar_train
}

# Generate experiments dictionary
experiments = {}

for seq_type, env_type in configurations:
    for loss_name, loss_fn in losses.items():
        for net_name, net_class in networks.items():
            config_key = (seq_type, env_type, loss_name, net_name)
            if seq_type == 'short-seq':
                lr = 0.001
            elif seq_type == 'long-seq':
                lr = 0.0001
            # print(seq_type, env_type)
            if env_type == 'easy-seq':
                state_size = state_sizes[seq_type]
                action_size = state_size
            elif env_type == 'cifar10-seq':
                # Logits for MNIST and CIFAR10 are 10-dimensional
                state_size = 10
                if seq_type == 'short-seq':
                    action_size = 6
                elif seq_type == 'long-seq':
                    action_size = 11
            hidden_size = 64 if seq_type == "short-seq" else 128
            model_path = f"../rnn_models2/{seq_type}_{env_type}_{loss_name}_{net_name}.pth"
            learning_curve_path = f"../learning_curves2/{seq_type}_{env_type}_{loss_name}_{net_name}.npy"
            train_function = train_functions[env_type]

            experiments[config_key] = {
                'q_network': net_class,
                'loss': loss_fn,
                'state_size': state_size,
                'action_size': action_size,
                'hidden_size': hidden_size,
                'capacity': 100000,
                'batch_size': 32,
                'lr': lr,
                'gamma': 0.99,
                'model_path': model_path,
                'learning_curve_path': learning_curve_path,
                'mode': 'train-from-zero',
                'train_function': train_function,
                'train_steps': 200
            }

# Print to verify
'''
cntr = 0
for key, value in experiments.items():
    print(key, value)
    print(cntr)
    cntr += 1
print(cntr)
'''
# Run experiments
'''

from q_networks import RNNQNetwork, LSTMQNetwork
from agents import DQNAgent
import torch.nn as nn
# , mnist_train_lstm, cifar_train_lstm
from train import vanilla_train, mnist_train, cifar_train, vanilla_train_lstm, mnist_train_lstm, cifar_train_lstm

# Define configurations and parameters
configurations = [
    ("short-seq", "easy-seq"),
    ("short-seq", "mnist-seq"),
    ("short-seq", "cifar-seq"),
    ("long-seq", "easy-seq"),
    ("long-seq", "mnist-seq"),
    ("long-seq", "cifar-seq")
]

networks = {
    "RNN": RNNQNetwork,
    "LSTM": LSTMQNetwork
}

losses = {
    "MSELoss": nn.MSELoss,
    "SmoothL1Loss": nn.SmoothL1Loss
}

state_sizes = {
    "short-seq": 6,
    "long-seq": 11
}

train_functions_rnn = {
    "easy-seq": vanilla_train,
    "mnist-seq": mnist_train,
    "cifar-seq": cifar_train
}

train_functions_lstm = {
    "easy-seq": vanilla_train_lstm,
    "mnist-seq": mnist_train_lstm,
    "cifar-seq": cifar_train_lstm
}

# Generate experiments dictionary
experiments = {}

for seq_type, env_type in configurations:
    for loss_name, loss_fn in losses.items():
        for net_name, net_class in networks.items():
            config_key = (seq_type, env_type, loss_name, net_name)
            state_size = state_sizes[seq_type]
            action_size = state_size
            hidden_size = 64 if seq_type == "short-seq" else 128
            model_path = f"{seq_type}_{env_type}_{loss_name}_{net_name}.pth"
            learning_curve_path = f"{seq_type}_{env_type}_{loss_name}_{net_name}.npy"

            if net_name == "RNN":
                train_function = train_functions_rnn[env_type]
            elif net_name == "LSTM":
                train_function = train_functions_lstm[env_type]

            experiments[config_key] = {
                'q_network': net_class,
                'loss': loss_fn,
                'state_size': state_size,
                'action_size': action_size,
                'hidden_size': hidden_size,
                'capacity': 100000,
                'batch_size': 32,
                'lr': 0.001,
                'gamma': 0.99,
                'model_path': model_path,
                'learning_curve_path': learning_curve_path,
                'mode': 'train-from-zero',
                'train_function': train_function,
                'train_steps': 50000
            }

# Print to verify
for key, value in experiments.items():
    print(key, value)
'''
