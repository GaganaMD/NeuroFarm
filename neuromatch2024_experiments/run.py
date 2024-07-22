from experiments import experiments


def run_experiment(experiment):
    train = experiment['train_function']
    train(experiment)


for key, value in experiments.items():
    print(key, value)
    run_experiment(value)
