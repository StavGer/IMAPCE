import numpy_indexed as npi
import collections
import numpy as np
import pandas
from numpy.linalg import inv
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import pyreadr
import hdbscan
from sklearn import mixture
from PIL import Image

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'cyan', 'purple', 'brown', 'pink', 'grey']


def load_dataset(dataset_name, **kwargs):
    """
    Loads the selected dataset

    :param dataset_name: a string with the name of dataset to be loaded

    """
    np.random.seed(500)
    if dataset_name == "adult":
        # reading from the file
        df_data = pandas.read_csv('data/uci_adult_data.csv')
        start_col = df_data.columns.get_loc('d_0')
        input_data = df_data.iloc[:, start_col:].values.astype(np.float32)
        gender_col = np.expand_dims(pandas.factorize(df_data['gender'])[0], 1)
        ethnicity_col = np.expand_dims(pandas.factorize(df_data['ethnicity'])[0], 1)
        income_col = np.expand_dims(pandas.factorize(df_data["income"])[0], 1)
        labels = np.concatenate((gender_col, ethnicity_col, income_col), axis=1)
        return input_data, labels

    elif dataset_name == "synthetic":
        data_df = pandas.read_csv("data/synthetic_dataset.csv")
        data_df.head()
        prior_col = "batch"
        label_col = "celltype"
        input_data = np.asarray(data_df.iloc[:, 3:13])
        prior_column = np.asarray(data_df[prior_col]).reshape(-1, 1)
        column_to_explore = np.asarray(data_df[label_col]).reshape(-1, 1)
        data_labels = np.concatenate((column_to_explore, prior_column), axis=1)
        return input_data, data_labels

    elif dataset_name == 'MNIST':
        inp = MNIST(".", train=True, download=True)
        inp_arr = inp.train_data.numpy()
        inp_f = np.reshape(inp_arr, (inp_arr.shape[0], inp_arr.shape[1]*inp_arr.shape[2]))  # reshape in a 2D array
        targets = inp.targets.numpy()
        if not kwargs:
            return inp_f, targets
        else:
            ind = np.arange(inp_f.shape[0])
            np.random.shuffle(ind)
            ind = ind[:kwargs['num_samples']]
            inp_f = inp_f[ind]
            targets = targets[ind]
            return inp_f, targets

    elif dataset_name == 'Fashion-MNIST':
        inp = FashionMNIST(".", train=True, download=True)
        inp_arr = inp.train_data.numpy()
        inp_f = np.reshape(inp_arr, (inp_arr.shape[0], inp_arr.shape[1]*inp_arr.shape[2]))  # reshape in a  2D array
        targets = inp.targets.numpy()
        str_targets = np.array(['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                                'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
        labels = np.empty(targets.shape[0], dtype=object)
        for i, l in enumerate(np.unique(targets)):
            id = np.where(targets == l)[0]
            labels[id] = str_targets[i]
        if not kwargs['num_samples']:
            return inp_f, labels
        else:
            ind = np.arange(inp_f.shape[0])
            np.random.shuffle(ind)
            ind = ind[:kwargs['num_samples']]
            inp_f = inp_f[ind]
            labels = labels[ind]
            return inp_f, labels

    elif dataset_name == "MNIST-FMNIST":
        inp_mnist = MNIST(".", train=True, download=True)  # load MNIST data
        inp_arr_mnist = inp_mnist.train_data.numpy()
        inp_fmnist = FashionMNIST(".", train=True, download=True)  # load Fashion-MNIST data
        inp_arr_fmnist = inp_fmnist.train_data.numpy()
        fmnist_labels = inp_fmnist.targets.numpy()
        str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                               'Sneaker', 'Bag', 'Ankle boot'])
        fmnist_str_labels = np.empty(fmnist_labels.shape[0], dtype=object)
        for i, l in enumerate(np.unique(fmnist_labels)):
            id = np.where(fmnist_labels == l)[0]
            fmnist_str_labels[id] = str_labels[i]
        input_data = np.zeros((kwargs['num_samples'], inp_arr_mnist.shape[1], inp_arr_mnist.shape[2]))
        ind = np.arange(input_data.shape[0])
        np.random.shuffle(ind)  # Randomly select a subset of the MNIST and Fashion-MNIST datasets
        ind = ind[:kwargs['num_samples']]
        for i in range(ind.shape[0]):
            input_data[i] = inp_arr_mnist[ind[i]] + inp_arr_fmnist[ind[i]]  # Superimpose MNIST with Fashion-MNIST
        input_data_transf = input_data.reshape((kwargs['num_samples'], input_data.shape[1] * input_data.shape[2]))
        fmnist_labels = fmnist_labels[ind]
        id_0 = np.where(str_labels == kwargs['class_0'])[0]
        id_1 = np.where(str_labels == kwargs['class_1'])[0]
        desired_labels = np.array([id_0, id_1])  # Select two classes of Fashion-MNIST using the index of str_labels
        ind = []
        for i, l in enumerate(desired_labels):
            id = np.where(fmnist_labels == l)[0]
            ind.append(id)
        ind = np.array(ind, dtype=object)
        ind = np.concatenate((ind[0], ind[1]))
        input_data_transf = input_data_transf[ind]
        fmnist_labels = fmnist_labels[ind]
        return input_data_transf, fmnist_labels
    elif dataset_name == 'UCI dataset':
        data_read = pyreadr.read_r('data/segment.rds')  # load from .rds file
        data_read = data_read[None]  # extract the pandas data frame for the only object available
        data_np = data_read.to_numpy()
        im_class = data_np[:, 0]
        input_data = np.delete(data_np, 0, 1)
        input_data = np.array(input_data, dtype=float)
        return input_data, im_class

    elif dataset_name == "CIFAR":
        data_pre_path = './cifar-100/'  # change this path
        # File paths
        data_train_path = data_pre_path + 'train'
        # Read dictionary
        data_train_dict = unpickle(data_train_path)
        # Get data (change the coarse_labels if you want to use the 100 classes)
        data_train = data_train_dict[b'data']
        fine_label_train = np.array(data_train_dict[b'fine_labels'])
        grey_image_cropped = np.zeros((kwargs['num_samples'], 28, 28))
        for i in range(kwargs['num_samples']):
            z = data_train[i].reshape(3, 32, 32)
            z = z.transpose(1, 2, 0)
            grey_image = np.array(Image.fromarray(z).convert('L'))
            grey_image_cropped[i] = grey_image[2:30, 2:30]
        grey_image_flattened = grey_image_cropped.reshape((kwargs['num_samples'],
                                                           grey_image_cropped.shape[1]*grey_image_cropped.shape[2]))
        return grey_image_flattened, fine_label_train[:kwargs['num_samples']]

    elif dataset_name == "CIFAR-FMNIST":
        data_pre_path = './cifar-100/'  # change this path
        # File paths
        data_train_path = data_pre_path + 'train'
        # Read dictionary
        data_train_dict = unpickle(data_train_path)
        # Get data (change the coarse_labels if you want to use the 100 classes)
        data_train = data_train_dict[b'data']
        grey_image_cropped = np.zeros((kwargs['num_samples'], 28, 28))
        for i in range(kwargs['num_samples']):
            z = data_train[i].reshape(3, 32, 32)
            z = z.transpose(1, 2, 0)
            grey_image = np.array(Image.fromarray(z).convert('L'))
            grey_image_cropped[i] = grey_image[2:30, 2:30]

        inp_fmnist = FashionMNIST(".", train=True, download=True)  # load Fashion-MNIST data
        inp_arr_fmnist = inp_fmnist.train_data.numpy()
        fmnist_labels = inp_fmnist.targets.numpy()
        str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
        fmnist_str_labels = np.empty(fmnist_labels.shape[0], dtype=object)
        for i, l in enumerate(np.unique(fmnist_labels)):
            id = np.where(fmnist_labels == l)[0]
            fmnist_str_labels[id] = str_labels[i]
        input_data = np.zeros((kwargs['num_samples'], inp_arr_fmnist.shape[1], inp_arr_fmnist.shape[2]))

        ind = np.arange(input_data.shape[0])
        np.random.shuffle(ind)  # Randomly select a subset of CIFAR-100 and Fashion-MNIST samples
        ind = ind[:kwargs['num_samples']]
        for i in range(ind.shape[0]):
            input_data[i] = grey_image_cropped[ind[i]] + inp_arr_fmnist[ind[i]]  # Superimpose CIFAR-100 with F-MNIST
        input_data_transf = input_data.reshape((kwargs['num_samples'], input_data.shape[1] * input_data.shape[2]))
        fmnist_labels = fmnist_labels[ind]
        id_0 = np.where(str_labels == kwargs['class_0'])[0]
        id_1 = np.where(str_labels == kwargs['class_1'])[0]
        desired_labels = np.array([id_0, id_1])  # Select a combination of classes of Fashion-MNIST data using the index of str_labels
        ind = []
        for i, l in enumerate(desired_labels):
            id = np.where(fmnist_labels == l)[0]
            ind.append(id)
        ind = np.array(ind, dtype=object)
        ind = np.concatenate((ind[0], ind[1]))
        input_data_transf = input_data_transf[ind]
        fmnist_labels = fmnist_labels[ind]
        return input_data_transf, fmnist_labels


def plot_clustering_results(root, iteration, proj, patterns_ind, non_patterns_ind,
                            predicted_labels, unexplored_data, non_patterns_labels, total_labels, colors):
    """
    Plot a triplet of subplots:
    i) data projection,
    ii) unexplored data projections clustered with DPGMM or HDBSCAN,
    iii) unexplored data projections coloured according to their true class

    :param root: path to save the triplet of subplots
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
    num_subplots = 3
    plt.subplot(num_subplots, 1, 1)  # Triplet i)
    if patterns_ind is None:  # all data are unexplored and have black colour
        a = plt.scatter(proj[:, 0], proj[:, 1], marker='.', color="black")
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    else:  # Black points remain unexplored while grey points are the already explored data
        plt.scatter(proj[non_patterns_ind, 0], proj[non_patterns_ind, 1], marker='.', color="black")
        a = plt.scatter(proj[patterns_ind, 0], proj[patterns_ind, 1], marker='.', color="grey", alpha=0.2)
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    plt.subplot(num_subplots, 1, 2)  # Triplet ii)
    for u_class in (np.unique(predicted_labels)):
        ind = np.where(predicted_labels == u_class)[0]
        a = plt.scatter(unexplored_data[ind, 0], unexplored_data[ind, 1], marker='.',
                        color=colors[u_class % len(colors)])
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
    plt.subplot(num_subplots, 1, 3)  # Triplet iii)
    for i, l in enumerate(np.unique(non_patterns_labels)):
        id = np.where(non_patterns_labels == l)[0]
        c = int(np.where(total_labels == l)[0])
        a = plt.scatter(unexplored_data[id, 0], unexplored_data[id, 1], marker='.', color=colors[c % len(colors)])
        xax = a.axes.get_xaxis()
        xax.set_visible(False)
        yax = a.axes.get_yaxis()
        yax.set_visible(False)
        plt.tight_layout()
    plt.ioff()
    plt.savefig(f'{root}//Proj{iteration}.pdf')  # save the triplet
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

    clusters_num = np.unique(labeled_points).shape[0]  # number of clusters
    if mean_matrix is None:  # hdbscan clustering
        outliers_index = np.where(labeled_points == -1)[0]
        labeled_points = np.delete(labeled_points, outliers_index)  # remove the outliers from the labels array
        mean_m = np.zeros((clusters_num, 2))
        Cluster_cov_matrix = np.zeros((2*clusters_num, 2))
        for i, l in enumerate(clusters_num):  # Mean and Cov matrices are computed for each cluster
            ind = np.where(labeled_points == l)[0]
            Cluster_cov_matrix[2*i, :] = np.cov(proj[ind, :].T)[0, :]
            Cluster_cov_matrix[2*i + 1, :] = np.cov(proj[ind, :].T)[1, :]
            m = np.mean(np.squeeze(proj[ind, :]), axis=0)
            mean_m[i, :] = m
    else:  # DPGMM clustering , mean and Cov matrices are computed automatically by a DPGMM
        mean_m = mean_matrix
    mahalanobis_dist = np.zeros((clusters_num, clusters_num))
    outliers_flag = np.zeros(clusters_num)
    for i in range(clusters_num):  # Find clusters with outliers (not acceptable size)
        cluster_size = np.where(labeled_points == np.unique(labeled_points)[i])[0].shape[0]  # size of cluster i
        if cluster_size < min_clust_size:
            outliers_flag[i] = 1
    for i in range(clusters_num):  # Compute the Mahalanobis distance between acceptable clusters
        for j in range(clusters_num):
            if i != j and outliers_flag[i] != 1 and outliers_flag[j] != 1:
                mahalanobis_dist[i, j] = np.sqrt((mean_m[i, :] - mean_m[j, :]).T@inv(Cluster_cov_matrix[2*j:2*j+2, :])
                                                 @ (mean_m[i, :] - mean_m[j, :]))
    mahalanobis_dist = (mahalanobis_dist + mahalanobis_dist.T)/2  # define the symmetric distance between two clusters
    clusters_val = np.unique(labeled_points)
    clusters_sorted_ind = np.argsort(-np.sum(mahalanobis_dist, axis=1))  # sort clusters w.r.t their distances
    clusters_sorted = clusters_val[clusters_sorted_ind]
    list_labels_clust = labeled_points.tolist()
    num_valid_clusters = np.where(outliers_flag == 0)[0].shape[0]  # number of acceptable clusters
    valid_clusters_id = np.where(outliers_flag == 0)[0]  # indices of acceptable clusters
    invalid_clusters_num = clusters_num - num_valid_clusters  # number of not acceptable clusters (they have outliers)
    if invalid_clusters_num == clusters_num:  # If all clusters are not acceptable, select none
        return mahalanobis_dist, None, outliers_flag
    elif num_valid_clusters == 2:  # If only two clusters have acceptable size, both are considered as most distinct
        return mahalanobis_dist, clusters_val[valid_clusters_id], outliers_flag
    else:  # Find most distinct cluster as the one with the greatest cumulative distance from the rest
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
        b = np.asarray(b)
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
        dpgmm = mixture.BayesianGaussianMixture(n_components=max_num_clusters,
                                                max_iter=100, n_init=3, random_state=20).fit(proj)
        m = dpgmm.means_
        c = dpgmm.covariances_
        cov_mat = np.reshape(c, (max_num_clusters*2, 2))
        labels_clust_dpgmm = dpgmm.predict(proj)
        return labels_clust_dpgmm, m, cov_mat
    elif clustering_algorithm == "HDBSCAN":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=15)
        hdbscan_labels = clusterer.fit_predict(proj)
        outliers_index = np.where(hdbscan_labels == -1)[0]
        if outliers_index.shape[0] == proj.shape[0]:
            min_cluster_size -= 10
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
            hdbscan_labels = clusterer.fit_predict(proj)
        m = None
        cov_mat = None
        return hdbscan_labels, m, cov_mat


def to_one_hot(input_data):
    one_hot_input = np.zeros((input_data.shape[0], 2*input_data.shape[1]))
    if input_data.shape[1] == 1:
        one_hot_input[:, 0] = input_data[:, 0]
        id0 = np.where(input_data[:, 0] == 0)
        one_hot_input[id0, 1] = 1
    else:
        id00 = np.intersect1d(np.where(input_data[:, 0] == 0)[0], np.where(input_data[:, 1] == 0)[0])
        id01 = np.intersect1d(np.where(input_data[:, 0] == 0)[0], np.where(input_data[:, 1] == 1)[0])
        id10 = np.intersect1d(np.where(input_data[:, 0] == 1)[0], np.where(input_data[:, 1] == 0)[0])
        id11 = np.intersect1d(np.where(input_data[:, 0] == 1)[0], np.where(input_data[:, 1] == 1)[0])
        one_hot_input[id00, 0] = 1
        one_hot_input[id01, 1] = 1
        one_hot_input[id10, 2] = 1
        one_hot_input[id11, 3] = 1
    return one_hot_input


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_embeddings(dataset_name, path, embs, labels):
    plt.clf()
    if dataset_name == "MNIST-FMNIST" or dataset_name == "CIFAR-FMNIST":
        str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                               'Sneaker', 'Bag', 'Ankle boot'])
        for i, l in enumerate(np.unique(labels)):
            id = np.where(labels == l)[0]
            a = plt.scatter(embs[id, 0], embs[id, 1],
                            marker='.', color=colors[i], label=str_labels[l], alpha=0.8)
        id1 = np.where(labels == 1)[0]
        id2 = np.where(labels == 3)[0]
        labels[id1] = 0
        labels[id2] = 1
        plt.legend(loc='best', prop={'size': 20})
    elif dataset_name == 'synthetic':
        prior_labels = labels[:, 1]
        unexplored_labels = labels[:, 0]
        marker = np.array(['o', '^', 's'])
        for i, l in enumerate(np.unique(unexplored_labels)):
            id = np.where(unexplored_labels == l)[0]
            for j, k in enumerate(np.unique(prior_labels)):
                id2 = np.where(prior_labels == k)[0]
                idt = np.intersect1d(id, id2)
                a = plt.scatter(embs[idt, 0], embs[idt, 1], marker=marker[j], color=colors[i],
                                facecolors='None', alpha=0.3)
        labels = ["Cluster 1 of dims 5-6", "Cluster 2 of dims 5-6", "Cluster 3 of dims 5-6"]
        f = lambda m, c, fill: plt.plot([],[], marker=m, color=c, ls="none", fillstyle=fill)[0]
        handles = [f("s", colors[i], 'full') for i in range(3)]
        handles += [f(marker[i], "k", 'none') for i in range(2)]
        labels += ["Cluster 1 of dims 1-4", "Cluster 2 of dims 1-4"]
        plt.legend(handles, labels, loc='best', framealpha=1)
    elif dataset_name == 'adult':
        markers = ['o', '^']
        marker_size = 20
        gender_col = labels[:, 0]
        ethnicity_col = labels[:, 1]
        income_col = labels[:, 2]
        for c in np.unique(gender_col):
            for m in np.unique(ethnicity_col):
                for f in np.unique(income_col):
                    mask = (gender_col == c) & (ethnicity_col == m) & (income_col == f)
                    facecolors = colors[c] if f else 'none'
                    edgecolors = 'none' if f else colors[c]
                    marker_size = 12 if m == 1 else marker_size
                    a = plt.scatter(embs[mask, 0], embs[mask, 1], marker_size,
                                    edgecolors=edgecolors, marker=markers[m], facecolors=facecolors,
                                    alpha=0.8, linewidth=0.6)
        labels = ["female", "male", "white", "other"]
        fill = ['full', 'none']
        f = lambda m, c, fill: plt.plot([], [], marker=m, color=c, ls="none", fillstyle=fill)[0]
        handles = [f("s", colors[i], 'full') for i in range(2)]
        handles += [f(markers[i], "k", 'none') for i in range(2)]
        handles += [f("s", "k", fill[i]) for i in range(2)]
        labels += ["income<50k", "income>50k"]
        plt.legend(handles, labels, loc='best', framealpha=1)
    else:   # for any other dataset with class labels
        for i, l in enumerate(np.unique(labels)):
            id = np.where(labels == l)[0]
            a = plt.scatter(embs[id, 0], embs[id, 1],
                            marker='.', color=colors[i], label=labels[l], alpha=0.8)
    xax = a.axes.get_xaxis()
    xax.set_visible(False)
    yax = a.axes.get_yaxis()
    yax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}{dataset_name}.pdf')
    return


def print_metrics(metrics, classifier_type):
    print(f'\n{classifier_type} Classifier Results:')
    for metric_name, values in metrics.items():
        print(f'Mean {metric_name.capitalize()} for {classifier_type} classifier: {np.mean(values):.4f}')
        print(f'Std {metric_name.capitalize()} for {classifier_type} classifier: {np.std(values):.4f}')


