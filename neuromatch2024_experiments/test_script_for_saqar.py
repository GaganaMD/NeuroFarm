from experiments import experiments
from agents import DQNAgent
from environments import DelaySampleToMatchEnv
from eval_scripts import prepare_data
import numpy as np
import matplotlib.pyplot as plt


def move_tuple_to_device(hidden_tuple, device):
    return (hidden_tuple[0].to(device), hidden_tuple[1].to(device))


experiment = experiments[('short-seq', 'cifar10-seq', 'MSELoss', 'RNN')]
env = DelaySampleToMatchEnv(n_stimuli=experiment['action_size'] - 1)
agent = DQNAgent(experiment)

agent.load_model('eval')

agent.epsilon = 0.00

print(agent.q_network)

n_episodes = 10
win_pct_list = []
scores = []
hids = []

train_data, class_dct = prepare_data()

for i in range(n_episodes):
    done = False
    state = env.reset()  # Reset the environment
    indices = class_dct[int(state)]
    random_index = np.random.choice(indices)
    state = train_data[random_index].flatten()
    score = 0
    if isinstance(hidden, tuple):
        hidden = move_tuple_to_device(hidden, agent.device)
    else:
        hidden = hidden.to(agent.device)
    while not done:
        action, next_hidden = agent.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)  # Take the action
        indices = class_dct[int(next_state)]
        random_index = np.random.choice(indices)
        next_state = train_data[random_index].flatten()
        hidden = next_hidden
        # hids.append(hidden)
        state = next_state  # Move to the next state
        score += reward
        if isinstance(hidden, tuple):
            hids.append(hidden.detach().cpu().numpy())
        else:
            hids.append(hidden[0].detach().cpu().numpy())


hids = np.array(hids).squeeze(1).squeeze(1)
print(hids.shape)
print(hids)
