import numpy as np
from funcs import load_dataset
from exploration import Explore

datasets = ["MNIST", "Fashion-MNIST", "UCI dataset", "MNIST-FMNIST", "synthetic", "adult", "CIFAR-FMNIST"]
dataset_name = datasets[-3] # select a dataset from the list of datasets

iterative = False
if dataset_name == 'MNIST-FMNIST' or dataset_name == 'CIFAR-FMNIST':
    num_samples = 30000 # for complex data
    class_0 = 'T-shirt/Top'  # user-selected
    class_1 = 'Dress'  # user-selected
    input_data, data_labels = load_dataset(dataset_name, num_samples=num_samples,
                                           class_0=class_0, class_1=class_1)
    if dataset_name == 'MNIST-FMNIST':
        background_samples, _ = load_dataset('MNIST', num_samples=1000)
    else:
        background_samples, _ = load_dataset('CIFAR', num_samples=5000)
elif dataset_name == 'synthetic':
    input_data, data_labels = load_dataset('synthetic')
    background_samples = 0*np.random.rand(input_data.shape[0], input_data.shape[1])
    background_samples[:, 0:4] = input_data[:, 0:4]  # features 1-4 are prior
elif dataset_name == 'adult':
    input_data, data_labels = load_dataset('adult')
    background_samples = 0*np.random.rand(input_data.shape[0], input_data.shape[1])
    prior_attr = 'gender'  # user-selected
    if prior_attr == 'ethnicity':
        background_samples[:, 2] = input_data[:, 2]
    elif prior_attr == 'gender':
        background_samples[:, 3] = input_data[:, 3]
    elif prior_attr == 'gender-ethnicity':
        background_samples[:, 2] = input_data[:, 2]
        background_samples[:, 3] = input_data[:, 3]
elif dataset_name == 'UCI dataset' or dataset_name == 'MNIST':
    iterative = True
    input_data, data_labels = load_dataset(dataset_name)
    background_samples = None

framework = Explore(model='IMAPCE',
                    dataset_name=dataset_name,
                    input_data=input_data,
                    data_labels=data_labels,
                    clustering_algorithm="DPGMM",
                    background_samples=background_samples,
                    alpha=1,
                    mu=100,
                    seed=5)
if iterative:
    framework.Iterative_exploration(min_cluster_size=75, max_clusters_num=5, max_exploration_iterations=25)
else:
    framework.run_exploration()
