from experiments import experiments

experiment = experiments[("short-seq", "easy-seq", "RNN", "Rew-1")]
experiment = experiments[("long-seq", "cifar10-seq", "LSTM", "Rew-10")]
experiment = experiments[("short-seq", "easy-seq", "RNN", "Rew-1")]


def run_experiment(experiment):
    train = experiment['train_function']
    train(experiment)


# for key, value in experiments.items():
    # print(key, value)
run_experiment(experiment)
