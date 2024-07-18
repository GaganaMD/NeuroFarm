import matplotlib.pyplot as plt
import numpy as np

dat = np.load(
    '/home/maria/NeuroFarm/learning_curves/short-seq_cifar10-seq_MSELoss_RNN.npy')
print(dat.shape)
plt.plot(dat)
plt.show()

dat = np.load('/home/maria/NeuroFarm/cifar10_task/logits/logits_cifar10.npy')
# dat = np.load('/home/maria/NeuroFarm/cifar10_task/logits/embeddings copy.npy')

plt.imshow(dat)
plt.show()
