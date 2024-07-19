from experiments import experiments
from agents import DQNAgent
from environments import DelaySampleToMatchEnv
from eval_scripts import prepare_data
import numpy as np
import matplotlib.pyplot as plt

experiment = experiments[('short-seq', 'cifar10-seq', 'MSELoss', 'RNN')]
env = DelaySampleToMatchEnv(n_stimuli=experiment['action_size'] - 1)
agent = DQNAgent(experiment)

agent.load_model('eval')

agent.epsilon = 0.00

print(agent.q_network)

n_episodes = 100
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
    hidden = agent.q_network.init_hidden().to(agent.device)
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
        hids.append(hidden.detach().cpu().numpy())


hids = np.array(hids).squeeze(1).squeeze(1)
print(hids.shape)
print(hids)


from sklearn.decomposition import PCA

pca = PCA(n_components=3, svd_solver='full')
# n_samples, n_features
print(hids.shape)
pcs = pca.fit_transform(hids)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize

# Create a repeating pattern of indices for colors
num_points = len(pcs)
cycle_length = 12
indices = np.arange(num_points) % cycle_length

# Normalize the indices to fit within the range of the colormap
norm = Normalize(vmin=0, vmax=cycle_length - 1)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2],
                     c=indices, cmap='bwr', norm=norm)
plt.colorbar(scatter, ticks=range(cycle_length))
plt.show()
