import numpy as np
import matplotlib.pyplot as plt

dat = np.load(
    '/home/maria/NeuroFarm/learning_curves2/short-seq_cifar10-seq_MSELoss_RNN.npy')
dat = np.load(
    '/home/maria/NeuroFarm/learning_curves2/short-seq_easy-seq_MSELoss_RNN.npy')
dat = np.load(
    '/home/maria/NeuroFarm/learning_curves2/short-seq_cifar10-seq_MSELoss_RNN.npy')
dat = np.load(
    '/home/maria/NeuroFarm/learning_curves2/long-seq_digits-seq_MSELoss_RNN.npy')
print(dat.shape)

rng = np.arange(0, dat.shape[0]) * 100
plt.plot(rng, dat)
plt.show()
plt.plot(dat)
