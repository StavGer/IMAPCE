import numpy.linalg
import numpy_indexed as npi
import collections
import numpy as np, h5py, os
import time, pandas
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from numpy.linalg import inv
from torchvision.datasets import MNIST, FashionMNIST
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import pyreadr, random, hdbscan, pymanopt, argparse, scipy
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
import autograd.numpy as autonp
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'cyan', 'purple', 'brown', 'pink', 'grey']


def load_dataset(dataset_name, n_samples):
    """
    Loads the selected dataset

    :param dataset_name: a string with the name of dataset to be loaded
    :param n_samples: if 'All' all data samples are loaded, else the number of randomly selected samples to be loaded

    """

    np.random.seed(500)
    if dataset_name == "adult":
        features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                    "Hours per week", "Country", "Target"]
        # reading from the file
        original_train = pandas.read_csv('adult.data', names=features, sep=r'\s*,\s*', engine='python', na_values="?")
        df_data = pandas.read_csv('uci_adult_data.csv')
        start_col = df_data.columns.get_loc('d_0')
        X = df_data.iloc[:, start_col:].values.astype(np.float32)
        gender_col = np.expand_dims(pandas.factorize(df_data['gender'])[0], 1)
        ethnicity_col = np.expand_dims(pandas.factorize(df_data['ethnicity'])[0], 1)
        income_col = np.expand_dims(pandas.factorize(df_data["income"])[0], 1)
        labels = np.concatenate((gender_col, ethnicity_col, income_col), axis = 1)
        return X, labels

    elif dataset_name == "synthetic":
        data_df = pandas.read_csv("synthetic_dataset.csv")
        data_df.head()
        prior_col = "batch"
        label_col = "celltype"
        X = np.asarray(data_df.iloc[:, 3:13])
        Y = np.asarray(data_df[prior_col])
        L = np.asarray(data_df[label_col])
        return X, L

    elif dataset_name == 'MNIST':
        inp = MNIST(".", train=True, download=True)
        inp_arr = inp.train_data.numpy()
        inp_f = np.reshape(inp_arr, (inp_arr.shape[0], inp_arr.shape[1]*inp_arr.shape[2])) # reshape the data in a 2D array
        targets = inp.targets.numpy()
        if n_samples == 'All':
            return inp_f, targets
        else:
            ind = np.arange(inp_f.shape[0])
            np.random.shuffle(ind)
            ind = ind[:n_samples]
            inp_f = inp_f[ind]
            targets = targets[ind]
            return inp_f, targets
    elif dataset_name == 'Fashion-MNIST':
        inp = FashionMNIST(".", train=True, download=True)
        inp_arr = inp.train_data.numpy()
        inp_f = np.reshape(inp_arr, (inp_arr.shape[0], inp_arr.shape[1]*inp_arr.shape[2])) # reshape the dataset in a 2D array
        targets = inp.targets.numpy()
        str_targets = np.array(['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
        labels = np.empty(targets.shape[0], dtype=object)
        for i, l in enumerate(np.unique(targets)):
            id = np.where(targets == l)[0]
            labels[id] = str_targets[i]
        if n_samples == 'All':
            return inp_f, labels
        else:
            ind = np.arange(inp_f.shape[0])
            np.random.shuffle(ind)
            ind = ind[:n_samples]
            inp_f = inp_f[ind]
            labels = labels[ind]
            return inp_f, labels
    elif dataset_name == "MNIST and FMNIST":
        inp_mnist = MNIST(".", train=True, download=True)  # load MNIST data
        inp_arr_mnist = inp_mnist.train_data.numpy()
        inp_fmnist = FashionMNIST(".", train=True, download=True)  # load Fashion-MNIST data
        inp_arr_fmnist = inp_fmnist.train_data.numpy()
        fmnist_labels = inp_fmnist.targets.numpy()
        str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
        fmnist_str_labels = np.empty(fmnist_labels.shape[0], dtype= object)
        for i,l in enumerate(np.unique(fmnist_labels)):
            id = np.where(fmnist_labels == l)[0]
            fmnist_str_labels[id] = str_labels[i]
        input_data = np.zeros((n_samples, inp_arr_mnist.shape[1], inp_arr_mnist.shape[2]))
        ind = np.arange(input_data.shape[0])
        np.random.shuffle(ind) # Randomly select a subset of the MNIST and Fashion-MNIST datasets
        ind = ind[:n_samples]
        for i in range(ind.shape[0]):
            input_data[i] = inp_arr_mnist[ind[i]] + inp_arr_fmnist[ind[i]] # Create superimposition of MNIST with Fashion-MNIST
        input_data_transf = input_data.reshape((n_samples, input_data.shape[1] * input_data.shape[2]))
        fmnist_labels = fmnist_labels[ind]
        fmnist_str_labels = fmnist_str_labels[ind]
        desired_labels = np.array([8, 9]) # Select a combination of classes of Fashion-MNIST data using the index of str_labels
        ind = []
        for i,l in enumerate(desired_labels):
            id = np.where(fmnist_labels == l)[0]
            ind.append(id)
        ind = np.array(ind, dtype = object)
        ind = np.concatenate((ind[0], ind[1]))
        input_data_transf = input_data_transf[ind]
        complex = inp_arr_mnist[0] + inp_arr_fmnist[ind[-50]]

        a = plt.imshow(inp_arr_mnist[0], cmap = 'gray')
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
        plt.tight_layout()
        plt.savefig('mnist.pdf')
        plt.clf()
        a = plt.imshow(inp_arr_fmnist[ind[-50]], cmap = 'gray')
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
        plt.tight_layout()
        plt.savefig('fmnist.pdf')
        plt.clf()
        a = plt.imshow(complex, cmap='gray')
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
        plt.tight_layout()
        plt.savefig('complex.pdf')
        fmnist_labels = fmnist_labels[ind]
        return input_data_transf, fmnist_labels
    elif dataset_name == 'UCI dataset':
        data_read = pyreadr.read_r('segment.rds') # load from .rds file
        _ = data_read.keys()# let's check what objects we got: there is only None
        data_read = data_read[None] # extract the pandas data frame for the only object available
        data_np = data_read.to_numpy()
        im_class = data_np[:, 0]
        input_data = np.delete(data_np, 0, 1)
        input_data = np.array(input_data, dtype = float)
        if n_samples == 'All':
            n_samples = input_data.shape[0]
        input_data = input_data[:n_samples]
        return input_data, im_class

def plot_clustering_results(root, multilabel, most_informative_cluster, iteration, proj, patterns_ind, non_patterns_ind, predicted_labels, unexplored_data, non_patterns_labels, total_labels, colors):
    """
    Plot a triplet of subplots:
    i) data projection,
    ii) unexplored data projections clustered with DPGMM or HDBSCAN,
    iii) unexplored data projections coloured according to their true class

    :param root: path to save the triplet of subplots
    :param most_informative_cluster: most distinct cluster computed by mahalanobis_distances_between_clusters
    :param iteration: iteration of the exploration process
    :param proj: 2D data projections
    :param patterns_ind: index of samples that belong to the background data
    :param non_patterns_ind: index of samples that have not been explored yet
    :param predicted_labels: clustering labels of unexplored data samples
    :param unexplored_data: 2D unexplored data projections
    :param non_patterns_labels: true labels of unexplored data samples
    :param total_labels: true classes of the data damples
    :param colors: list of colors

    """

    plt.clf()
    if multilabel == False:
        num_subplots = 3
    else:
        num_subplots = 2
    plt.subplot(num_subplots, 1, 1)  # Triplet i)
    if patterns_ind is None: # all data are unexplored and have black colour
        a = plt.scatter(proj[:, 0], proj[:, 1], marker='.', color="black")
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    else: # Black points remain unexplored while grey points are the already explored data
        plt.scatter(proj[non_patterns_ind, 0], proj[non_patterns_ind, 1], marker='.', color="black")
        a = plt.scatter(proj[patterns_ind, 0], proj[patterns_ind, 1], marker='.', color="grey", alpha=0.2)
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    plt.subplot(num_subplots, 1, 2) # Triplet ii)
    for l in (np.unique(predicted_labels)):
        ind = np.where(predicted_labels == l)[0]
        a = plt.scatter(unexplored_data[ind, 0], unexplored_data[ind, 1], marker='.', color = colors[l % len(colors)])
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    if multilabel == False:
        plt.subplot(num_subplots, 1, 3) # Triplet iii)
        for i, l in enumerate(np.unique(non_patterns_labels)):
            id = np.where(non_patterns_labels == l)[0]
            c = int(np.where(total_labels == l)[0])
            a = plt.scatter(unexplored_data[id, 0], unexplored_data[id, 1], marker='.', color = colors[c % len(colors)])
            xax = a.axes.get_xaxis()
            xax.set_visible(False)
            yax = a.axes.get_yaxis()
            yax.set_visible(False)
            plt.tight_layout()
    plt.ioff()
    plt.savefig(root + 'Proj' + str(iteration) + '.pdf')  # save the triplet
    return


def mahalanobis_distances_between_clusters(labeled_points, proj, mean_matrix, Cluster_cov_matrix, min_clust_size):
    """
    Computes the distance between clusters with acceptable size

    :param labeled_points: vector of labels of points computed by the clustering algorithm
    :param proj: 2D data projections
    :param mean_matrix: matrix containing the means of all clusters
    :param Cluster_cov_matrix: matrix containing the covariance matrices of all clusters
    :param min_clust_size: minimum acceptable size of a cluster

    """

    clusters_num = np.unique(labeled_points).shape[0] # number of clusters
    if mean_matrix is None: # hdbscan clustering
        outliers_index = np.where(labeled_points == -1)[0]
        labeled_points = np.delete(labeled_points, outliers_index) # remove the outliers from the labels array
        mean_m = np.zeros((clusters_num, 2))
        Cluster_cov_matrix = np.zeros((2*clusters_num, 2))
        for i,l in enumerate(clusters_num):  # Mean and Cov matrices are computed for each cluster
            ind = np.where(labeled_points == l)[0]
            Cluster_cov_matrix[2*i, :] = np.cov(proj[ind, :].T)[0, :]
            Cluster_cov_matrix[2*i + 1, :] = np.cov(proj[ind, :].T)[1, :]
            m = np.mean(np.squeeze(proj[ind, :]), axis=0)
            mean_m[i, :] = m
    else:  # DPGMM clustering , mean and Cov matrices are computed automatically by DPGMM
        mean_m = mean_matrix
    mahalanobis_dist = np.zeros((clusters_num, clusters_num))
    outliers_flag = np.zeros(clusters_num)
    for i in range(clusters_num):  # Find clusters with outliers (not acceptable size)
        cluster_size = np.where(labeled_points == np.unique(labeled_points)[i])[0].shape[0]  # find the size of cluster i
        if cluster_size < min_clust_size:
            outliers_flag[i] = 1
    for i in range(clusters_num):  # Compute the Mahalanobis distance between acceptable clusters
        for j in range(clusters_num):
            if i != j and outliers_flag[i] != 1 and outliers_flag[j] != 1:
                mahalanobis_dist[i, j] = np.sqrt((mean_m[i, :] - mean_m[j, :]).T@inv(Cluster_cov_matrix[2*j:2*j+2, :])@(mean_m[i, :] - mean_m[j, :]))
    mahalanobis_dist = (mahalanobis_dist + mahalanobis_dist.T)/2  # define the symmetric distance between two clusters
    clusters_val = np.unique(labeled_points)
    clusters_sorted_ind = np.argsort(-np.sum(mahalanobis_dist, axis=1))  # sort clusters according to their cumulative symemtric distances from the others
    clusters_sorted = clusters_val[clusters_sorted_ind]
    list_labels_clust= labeled_points.tolist()
    num_valid_clusters = np.where(outliers_flag == 0)[0].shape[0]  # number of acceptable clusters
    valid_clusters_id = np.where(outliers_flag == 0)[0]  # indices of acceptable clusters
    invalid_clusters_num = clusters_num - num_valid_clusters  # number of not acceptable clusters(they have outliers)
    if invalid_clusters_num == clusters_num:  # If all clusters are not acceptable, no cluster is selected as most distinct
        return mahalanobis_dist, None, outliers_flag
    elif num_valid_clusters == 2:  # If only two clusters have acceptable size, both are considered as most distinct
        return mahalanobis_dist, clusters_val[valid_clusters_id], outliers_flag
    else:  # Select as most distinct the acceptable cluster with the highest cumulative symemtric distance from all others
        for i in range(clusters_num):
            clust_size = list_labels_clust.count(clusters_sorted[i])
            if clust_size > min_clust_size:
                most_informative_cluster = clusters_sorted[i]
                break
        return mahalanobis_dist, most_informative_cluster, outliers_flag


def common_samples_ind(array_1, array_2):
    """

    Finds the common samples between array_1 and array_2 and returns the index of such samples with respect to array_1

    """
    result = npi.indices(array_1, array_2)
    a = np.unique(result)
    if a.shape[0] == result.shape[0]:
        return result
    else:
        b = [item for item, count in collections.Counter(list(result)).items() if count > 1]
        b = np.asarray((b))
        new_ind = []
        for i in range(b.shape[0]):
            duplicate_id = np.where((result == b[i]))[0]
            for k in range(duplicate_id.shape[0] - 1):
                n = np.arange(result[duplicate_id[k]] + 1, array_1.shape[0])
                if k > 0:
                    n = np.arange(j + 1, array_1.shape[0])
                for j in n:
                    if (array_1[j, :] == array_1[result[duplicate_id[k]], :]).all():
                        new_ind.append(j)
                        break
        res = np.concatenate((a, new_ind))
        return res


def sum_(X, axis=0):

    s = np.sum(X, axis=axis, keepdims=True)
    if max(s.shape) == 1:
        return float(s)
    else:
        return s


def diag_(X):
    # diag returns (n,) array, this converts to (n,1)
    return np.expand_dims(np.diag(X), axis=0).T


def standardize_data(flag, data):
    """
    Mean-centering and standardization of data"

    flag : if False mean center the data, if True standardize the data

    """
    result = dict()
    for key, value in data.items():
        result[key] = value - np.mean(value, axis=0)  # Mean-centering the data
    if not flag:
        return result
    else:  # Standardizing the data
        for key, value in data.items():
            result[key] = (value - np.mean(value, axis=0)) / np.std(value, axis=0)
            result[key] = np.nan_to_num(result[key])
        return result


def clustering(clustering_algorithm, proj, min_cluster_size, max_num_clusters):
    """
    Perform the clustering using a selected algorithm and hyperparameters

    :param clustering_algorithm: "HDBSCAN" or "DPGMM"
    :param proj: 2D data projections
    :param min_cluster_size: minimum number of samples belonging to an acceptable cluster (required for HDBSCAN)
    :param max_num_clusters: maximum number of clusters (required for DPGMM)

    """

    if clustering_algorithm == "DPGMM":
        dpgmm = mixture.BayesianGaussianMixture(n_components=max_num_clusters, max_iter=100, n_init=3, random_state=20).fit(proj)
        m = dpgmm.means_
        c = dpgmm.covariances_
        cov_mat = np.reshape(c, (max_num_clusters*2,2))
        labels_clust_dpgmm = dpgmm.predict(proj)
        return labels_clust_dpgmm, m, cov_mat
    elif clustering_algorithm == "HDBSCAN":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=15)
        hdbscan_labels = clusterer.fit_predict(proj)
        outliers_index = np.where(hdbscan_labels == -1)[0] #
        if outliers_index.shape[0] == proj.shape[0]:
            min_cluster_size -= 10
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
            hdbscan_labels = clusterer.fit_predict(proj)
        m = None
        cov_mat = None
        return hdbscan_labels, m, cov_mat

def _parse_arguments(name, backends):
    parser = argparse.ArgumentParser(name)
    parser.add_argument(
        "-b",
        "--backend",
        help="backend to run the test on",
        choices=backends,
        default=backends[0],
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    return vars(parser.parse_args())

class ExampleRunner:
    def __init__(self, run_function, name, backends):
        self._arguments = _parse_arguments(name, backends)
        self._run_function = run_function
        self._name = name

    def run(self):
        backend = self._arguments["backend"]
        quiet = self._arguments["quiet"]
        if not quiet:
            print(self._name)
            print("-" * len(self._name))
            print(f"Using '{backend}' backend")
            print()
        self._run_function(backend=backend, quiet=quiet)

def create_cost_and_derivates(manifold, input_data, backend, alpha, mu, bg_data, unexplored_data, method):
    """
    Defines cost and derivatives for the optimization on the selected manifold

    :param manifold: the manifold(Stiefel is default) on which the optimization takes place
    :param input_data: original data samples with n x d dimensions
    :param backend: "autograd" or "numpy"
    :param alpha: hyperparameter controlling the trade-off of original data and background data variance
    :param mu: scaling parameter to make PCA(cPCA) and kurtosis terms comparable
    :param bg_data: the background data samples with m x d dimensions
    :param unexplored_data: the data samples that remain unexplored
    :param method: "PCA + kPP" if there is no background data, "cPCA + kPP" if there is background data

    """

    euclidean_gradient = euclidean_hessian = None
    if backend == "autograd":  # euclidean gradients are computed using Finite-Differences
        if method == "PCA + kPP":
            @pymanopt.function.autograd(manifold)
            def cost(w):
                s1 = 0
                for i in range(unexplored_data.shape[0]):
                    xi = np.expand_dims(unexplored_data[i, :], 1)
                    s1 += (autonp.trace(autonp.linalg.inv(w.T@unexplored_data.T@unexplored_data@w)@w.T@xi@xi.T@w)) ** 2
                return autonp.linalg.norm(input_data - input_data @ w @ w.T) ** 2 + mu * unexplored_data.shape[0] * s1
        elif method == "cPCA + kPP":
            @pymanopt.function.autograd(manifold)
            def cost(w):
                s1 = 0
                for i in range(unexplored_data.shape[0]):
                    xi = np.expand_dims(unexplored_data[i, :], 1)
                    s1 += (autonp.trace(autonp.linalg.inv(w.T@unexplored_data.T@unexplored_data@w)@w.T@xi@xi.T@w)) ** 2
                return autonp.linalg.norm(input_data - input_data @ w @ w.T) ** 2 - alpha * autonp.linalg.norm(bg_data - bg_data @ w @ w.T) ** 2 + mu * unexplored_data.shape[0] * s1

    elif backend == "numpy":  # Cost and Euclidean Gradient are explicitely provided

        if method == "PCA + kPP":
            @pymanopt.function.numpy(manifold)
            def cost(w):
                l1 = np.linalg.norm(input_data - input_data @ w @ w.T) ** 2
                l2 = mu * input_data.shape[0] * sum_(diag_((input_data @ w) @ np.linalg.inv((input_data @ w).T @ (input_data @ w)) @ (input_data @ w).T) ** 2)
                return l1 + l2

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(w):
                # w is the d x k projection matrix with respect to which we optimise the objective function
                A = w.T @ input_data.T @ input_data @ w
                Ainv = np.linalg.inv(A)
                scal = sqrtm(Ainv) @ w.T @ input_data.T
                scal = np.sqrt(sum_(scal ** 2))
                Mat = (np.ones((input_data.shape[1], 1)) @ scal) * input_data.T
                Mat2 = (np.ones((2, 1)) @ scal) * (w.T @ input_data.T)
                Mat2 = Mat2 @ Mat2.T
                Mat = Mat @ Mat.T
                gr = Mat @ w @ Ainv - (input_data.T @ input_data) @ w @ Ainv @ Mat2 @ Ainv
                gr = 4 * mu * input_data.shape[0] * gr
                return (
                        -2
                        * (input_data.T @ (input_data - input_data @ w @ w.T)
                           + (input_data - input_data @ w @ w.T).T @ input_data
                           ) @ w + gr)

        elif method == "cPCA + kPP":
            @pymanopt.function.numpy(manifold)
            def cost(w):
                return np.linalg.norm(input_data - input_data @ w @ w.T) ** 2 - alpha * np.linalg.norm(bg_data - bg_data @ w @ w.T) ** 2 + mu * unexplored_data.shape[0] * sum_(diag_((unexplored_data @ w) @ np.linalg.inv((unexplored_data @ w).T @ (unexplored_data @ w)) @ (unexplored_data @ w).T) ** 2)

            @pymanopt.function.numpy(manifold)
            def euclidean_gradient(w):
                A = w.T @ unexplored_data.T @ unexplored_data @ w
                Ainv = np.linalg.inv(A)
                scal = sqrtm(Ainv) @ w.T @ unexplored_data.T
                scal = np.sqrt(sum_(scal ** 2))
                Mat = (np.ones((input_data.shape[1], 1)) @ scal) * unexplored_data.T
                Mat2 = (np.ones((2, 1)) @ scal) * (w.T @ unexplored_data.T)
                Mat2 = Mat2 @ Mat2.T
                Mat = Mat @ Mat.T
                gr = Mat @ w @ Ainv - (unexplored_data.T@unexplored_data)@w@Ainv@Mat2@Ainv
                kurt_grad = 4 * mu * unexplored_data.shape[0] * gr
                return (
                        -2
                        * (input_data.T @ (input_data - input_data @ w @ w.T)
                           + (input_data - input_data @ w @ w.T).T @ input_data
                           ) @ w + 2 * alpha * (bg_data.T @ (bg_data - bg_data @ w @ w.T)
                                                + (bg_data - bg_data @ w @ w.T).T @ bg_data
                                                ) @ w + kurt_grad)

    return cost, euclidean_gradient, euclidean_hessian


def CalculateProjection(root, method, input_data, background_data, unexplored_data, alpha, mu, iteration, seed):
    """

    :param root: path to save the results
    :param method: "PCA + kPP" if there is no background data, "cPCA + kPP" if there is background data
    :param input_data: original data samples with n x d dimensions
    :param background_data: the background data samples with m x d dimensions
    :param unexplored_data: the data samples that remain unexplored
    :param alpha: hyperparameter controlling the trade-off of original data and background data variance
    :param mu: scaling parameter to make PCA(cPCA) and kurtosis terms comparable
    :param iteration: iteration of the exploration process
    :param seed: seed selected for reproducibility

    """

    SUPPORTED_BACKENDS = ("autograd", "numpy")
    backend = "numpy"
    np.random.seed(seed)
    d = input_data.shape[1]
    k = 2  # dimension of the low-dimensional representation
    input_data -= input_data.mean(axis=0)
    manifold = Stiefel(d, k)  # define the manifold for optimization
    t0 = time.time()
    cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(manifold, input_data, backend=backend,
                                                                            alpha=alpha, mu=mu,
                                                                            bg_data=background_data,
                                                                            unexplored_data=unexplored_data,
                                                                            method=method)  # Calculate cost and derivatives
    problem = pymanopt.Problem(
                    manifold,
                    cost,
                    euclidean_gradient=euclidean_gradient,
                    euclidean_hessian=euclidean_hessian,
                )  # Initialize the setup on PyManopt
    x0 = np.random.normal(size=(input_data.shape[1], k)) # Set initial point for the optimization
    optimizer = SteepestDescent(root=root, method=method, input_data=input_data, bg=background_data, alpha=alpha, mu=mu,
                                unexplored_data=unexplored_data, explore_iteration=iteration,
                                verbosity=2)  # Initialize the Steepest Descent Optimizer of PyManopt
    estimated_proj_matrix = optimizer.run(problem, initial_point=x0).point  # Run the optimization and calculate the projection matrix
    t1 = time.time() - t0
    print("Calculating the projection took " + str(t1) + " seconds")
    embeddings = input_data@estimated_proj_matrix  # compute data projection
    return embeddings, estimated_proj_matrix


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):

        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):

        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def to_one_hot(input):
    one_hot_input = np.zeros((input.shape[0], 2*input.shape[1]))
    if input.shape[1] == 1:
        one_hot_input[:, 0] = input[:, 0]
        id0 = np.where(input[:, 0] == 0)
        one_hot_input[id0, 1] = 1
    else:
        id00 = np.intersect1d(np.where(input[:, 0] == 0)[0], np.where(input[:, 1] == 0)[0])
        id01 = np.intersect1d(np.where(input[:, 0] == 0)[0], np.where(input[:, 1] == 1)[0])
        id10 = np.intersect1d(np.where(input[:, 0] == 1)[0], np.where(input[:, 1] == 0)[0])
        id11 = np.intersect1d(np.where(input[:, 0] == 1)[0], np.where(input[:, 1] == 1)[0])
        one_hot_input[id00, 0] = 1
        one_hot_input[id01, 1] = 1
        one_hot_input[id10, 2] = 1
        one_hot_input[id11, 3] = 1
    return one_hot_input

