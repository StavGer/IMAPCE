import numpy as np
import matplotlib.pyplot as plt
from funcs import load_dataset
from IMAPCE import IMAPCE
from Baselines import Baselines

plt.rcParams['figure.dpi'] = 100


datasets = ["MNIST", "Fashion-MNIST", "BNC-2007", "UCI dataset", "MNIST and FMNIST"]
dataset_name = datasets[3] # select a dataset from the list of datasets
colors=['red', 'green', 'blue', 'yellow', 'orange', 'cyan', 'purple', 'brown', 'pink', 'grey']


num_samples = 'All'
# input_data, labels_data = load_dataset(dataset_name, None) # load a data to define its background samples
# patterns_ind = np.where(labels_data == 'conversation')[0] # example of background class in BNC-2007 dataset
# patterns_ind = np.where(labels_data == 'FOLIAGE')[0] # example of background class in UCI dataset
# background_samples = input_data[patterns_ind] # defining the background samples in case there is background knowledge
background_samples = None  # assuming no prior knowledge
# background_samples, _ = load_dataset('MNIST', 1000)  # background data for MNIST and FMNIST data


framework = IMAPCE(dataset_name = dataset_name,
                                  num_samples = num_samples,
                                  selection = "automatic",
                                  clustering_algorithm = "DPGMM",
                                  min_cluster_size = 75,
                                  max_clusters_num = 5,
                                  max_exploration_iterations = 15,
                                  exploration = True,
                                  background_samples = background_samples,
                                  alpha = 1,
                                  mu = 1e5,
                                  seed = 45)
framework.Explore()

# Baselines_methods = ["original cPCA", "Boosted cPCA"]
# method = Baselines_methods[0] # select a Baseline from Baselines_methods

# framework = Baselines(dataset_name = dataset_name,
#                       method = method,
#                       num_samples = num_samples,
#                       num_alphas = 40,
#                       selection = "automatic",
#                       clustering_algorithm = "DPGMM",
#                       min_cluster_size = 75,
#                       max_clusters_num = 5,
#                       max_exploration_iterations = 15,
#                       exploration = True,
#                       background_samples = background_samples)

# framework.Explore()

