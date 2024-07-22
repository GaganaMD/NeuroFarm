import numpy as np
import torch
import time


def prepare_data():
    train_data = np.load(
        # '../cifar10_task/logits/embeddings copy.npy')
        '../cifar10_task/logits/logits_cifar10.npy')
    train_labels = np.load(
        # '../cifar10_task/logits/labels copy.npy')
        '../cifar10_task/logits/labels_cifar10.npy')

    # Adjust train_labels as per the original intent (adding 1)
    train_labels = train_labels + 1

    # Create a row of zeros with the same number of columns as train_data
    zeros_row = np.zeros((1, train_data.shape[1],))
    # print(train_data.shape, zeros_row.shape)
    train_data = torch.tensor(
        np.vstack((zeros_row, train_data)), dtype=torch.float32).to(device='cuda')

    # Append the label corresponding to the zeros row
    train_labels = np.hstack((0, train_labels))

    # print("Shape of train_labels_with_zeros:", train_labels.shape)
    # print("Shape of train_data_with_zeros:", train_data.shape)

    # Create a dictionary to store indices of each class
    class_dct = {}
    for i in range(12):  # Adjusted to iterate from 0 to 10 (inclusive)
        class_dct[i] = np.where(train_labels == i)[0]

    # Print example usage of class_dct
    # print("Indices of class 0:", class_dct[0])

    return train_data, class_dct


def cifar_eval(config_dict):
    def prepare_data():
        train_data = np.load(
            # '../cifar10_task/logits/embeddings copy.npy')
            '../cifar10_task/logits/logits_cifar10.npy')
        train_labels = np.load(
            # '../cifar10_task/logits/labels copy.npy')
            '../cifar10_task/logits/labels_cifar10.npy')

        # Adjust train_labels as per the original intent (adding 1)
        train_labels = train_labels + 1

        # Create a row of zeros with the same number of columns as train_data
        zeros_row = np.zeros((1, train_data.shape[1],))
        # print(train_data.shape, zeros_row.shape)
        train_data = torch.tensor(
            np.vstack((zeros_row, train_data)), dtype=torch.float32).to(device='cuda')

        # Append the label corresponding to the zeros row
        train_labels = np.hstack((0, train_labels))

        # print("Shape of train_labels_with_zeros:", train_labels.shape)
        # print("Shape of train_data_with_zeros:", train_data.shape)

        # Create a dictionary to store indices of each class
        class_dct = {}
        for i in range(12):  # Adjusted to iterate from 0 to 10 (inclusive)
            class_dct[i] = np.where(train_labels == i)[0]

        # Print example usage of class_dct
        # print("Indices of class 0:", class_dct[0])

        return train_data, class_dct

    train_data, class_dct = prepare_data()

    start = time.time()
    # n_stimuli = config_dict['action_size'] - 1
    # print('boom!', n_stimuli)
    n_stimuli = config_dict['action_size'] - 1
    # print('boom!', n_stimuli)
    env = DelaySampleToMatchEnv(n_stimuli=n_stimuli)
    if config_dict['q_network'] == RNNQNetwork:
        agent = DQNAgent(config_dict)
    elif config_dict['q_network'] == LSTMQNetwork:
        agent = LSTMDQNAgent(config_dict)
