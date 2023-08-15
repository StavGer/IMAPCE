import numpy as np
import matplotlib.pyplot as plt
from funcs import load_dataset
from Evaluation import Laplacian_scores
from IMAPCE import IMAPCE
from ContrastivePCA import Baselines
plt.rcParams['figure.dpi'] = 100



datasets = ["MNIST", "Fashion-MNIST", "UCI dataset", "MNIST and FMNIST", "synthetic", "adult"]
dataset_name = datasets[-1] # select a dataset from the list of datasets

num_samples = 30000
# num_samples = 'All'

input_data, labels_data = load_dataset(dataset_name, num_samples) # load a data to define its background samples
bg_data = 0*np.random.rand(input_data.shape[0], input_data.shape[1])

# synthetic data prior
# bg_data[:,0:4] = input_data[:, 0:4] #Synthetic dims 1-4


#UCI adult data priors
# bg_data[:, 2] = input_data[:, 2] # ethnicity
# bg_data[:, 3] = input_data[:, 3] #gender

# prior samples of MNIST with class '3'
# patterns_ind = np.where(labels_data == 3)[0]

# prior samples of MNIST with class 'FOLIAGE'
# patterns_ind = np.where(labels_data == 'FOLIAGE')[0] # example of background class in UCI dataset

# background_samples = input_data[patterns_ind]
background_samples = bg_data # assuming some prior knowledge

# if there are no prior data
# background_samples = None


# background data for MNIST and FMNIST data
# background_samples, _ = load_dataset('MNIST', 1000)

#
#
framework = IMAPCE(dataset_name = dataset_name,
                                                    num_samples = num_samples,
                                                    selection = "automatic",
                                                    clustering_algorithm = "DPGMM",
                                                    min_cluster_size = 400,
                                                    max_clusters_num = 5,
                                                    max_exploration_iterations = 25,
                                                    exploration = False,
                                                    background_samples = background_samples,
                                                    alpha = 1,
                                                    mu = 250,
                                                    seed = 5)
framework.Explore()

method = "original cPCA" # select a Baseline from Baselines_methods

framework = Baselines(dataset_name = dataset_name,
                      method = method,
                      num_samples = num_samples,
                      num_alphas = 40,
                      selection = "automatic",
                      clustering_algorithm = "DPGMM",
                      min_cluster_size = 400,
                      max_clusters_num = 5,
                      max_exploration_iterations = 25,
                      exploration = False,
                      background_samples = background_samples)

framework.Explore()
dataset_name = "synthetic"
labels = ''
# if dataset is synthetic or adult then the Laplacian scores plots are obtained by:
# labels = 'ethnicity' or 'gender' or 'gender-ethnicity'
Laplacian_scores(dataset_name, labels)