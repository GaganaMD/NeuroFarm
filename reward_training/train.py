from environments import DelaySampleToMatchEnv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromatch2024_experiments.q_networks import RNNQNetwork, LSTMQNetwork, GRUQNetwork
from neuromatch2024_experiments.agents import DQNAgent, LSTMDQNAgent
import numpy as np
import torch
import torch.nn.functional as F
import time


def move_tuple_to_device(hidden_tuple, device):
    return (hidden_tuple[0].to(device), hidden_tuple[1].to(device))


def vanilla_train(config_dict):
    start = time.time()
    n_stimuli = config_dict['state_size'] - 1
    reward = config_dict['reward']

    env = DelaySampleToMatchEnv(reward=reward, n_stimuli=n_stimuli)
    if config_dict['q_network'] == RNNQNetwork:
        agent = DQNAgent(config_dict)
    elif config_dict['q_network'] == LSTMQNetwork:
        agent = LSTMDQNAgent(config_dict)

    n_episodes = config_dict['train_steps']
    episode_rewards = []
    scores = []

    # Training loop
    for i in range(n_episodes):
        state = env.reset()  # Reset the environment
        state = F.one_hot(torch.tensor(state),
                          env.observation_space.n).to(dtype=torch.float32, device=agent.device)
        done = False
        score = 0
        hidden = agent.q_network.init_hidden()
        if isinstance(hidden, tuple):
            hidden = move_tuple_to_device(hidden, agent.device)
        else:
            hidden = hidden.to(agent.device)

        while not done:
            action, next_hidden = agent.select_action(state, hidden)
            next_state, reward, done, info = env.step(
                action)  # Take the action
            next_state = F.one_hot(torch.tensor(next_state),
                                   env.observation_space.n).to(dtype=torch.float32, device=agent.device)
            # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
            agent.store_transition(state, action, next_state,
                                   reward, hidden, next_hidden, done)
            agent.learn()  # Update Q-network
            hidden = next_hidden
            state = next_state  # Move to the next state
            score += reward

        scores.append(score)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            episode_rewards.append(avg_score)
            print(f"Episode {i} - Average Score: {avg_score:.2f}")

    end = time.time()
    agent.save_model()
    print(f'It took {end-start} seconds to train the model')
    np.save(config_dict['learning_curve_path'], episode_rewards)


def cifar_train(config_dict):
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
        for i in range(11):  # Adjusted to iterate from 0 to 10 (inclusive)
            class_dct[i] = np.where(train_labels == i)[0]

        # Print example usage of class_dct
        # print("Indices of class 0:", class_dct[0])

        return train_data, class_dct

    train_data, class_dct = prepare_data()

    start = time.time()
    # n_stimuli = config_dict['action_size'] - 1
    # print('boom!', n_stimuli)
    n_stimuli = config_dict['action_size'] - 1
    env = DelaySampleToMatchEnv(reward=reward, n_stimuli=n_stimuli)
    if config_dict['q_network'] == RNNQNetwork:
        agent = DQNAgent(config_dict)
    elif config_dict['q_network'] == LSTMQNetwork:
        agent = LSTMDQNAgent(config_dict)

    n_episodes = config_dict['train_steps']
    episode_rewards = []
    scores = []

    for i in range(n_episodes):
        done = False
        state = env.reset()  # Reset the environment
        indices = class_dct[int(state)]
        random_index = np.random.choice(indices)
        state = train_data[random_index].flatten()
        score = 0
        hidden = agent.q_network.init_hidden()
        if isinstance(hidden, tuple):
            hidden = move_tuple_to_device(hidden, agent.device)
        else:
            hidden = hidden.to(agent.device)
        while not done:
            action, next_hidden = agent.select_action(state, hidden)
            next_state, reward, done, info = env.step(
                action)  # Take the action
            indices = class_dct[int(next_state)]
            random_index = np.random.choice(indices)
            next_state = train_data[random_index].flatten()
            # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
            agent.store_transition(state, action, next_state,
                                   reward, hidden, next_hidden, done)
            agent.learn()  # Update Q-network
            hidden = next_hidden
            state = next_state  # Move to the next state
            score += reward
        scores.append(score)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            episode_rewards.append(avg_score)
            print(f"Episode {i} - Average Score: {avg_score:.2f}")

    end = time.time()
    agent.save_model()
    print(f'It took {end-start} seconds to train the model')
    np.save(config_dict['learning_curve_path'], episode_rewards)


def mnist_train(config_dict):
    pass
