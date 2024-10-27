import os
import numpy as np
from funcs import standardize_data, common_samples_ind, clustering, plot_embeddings, print_metrics
from funcs import mahalanobis_distances_between_clusters, plot_clustering_results
from Evaluation import Evaluation_iterative, evaluate_classifier, Evaluate_NLS
from methods import CalculateProjection
import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'cyan',
          'purple', 'brown', 'pink', 'grey', 'magenta', 'gold', 'indigo']
plt.rcParams.update({'font.size': 12})


class Explore:
    """
    Informative MAnifold Projections for Cluster Exploration

    Parameters:
    --------
    dataset_name : string of the dataset's name
    num_samples : the number of samples of the selected dataset,
                  "All" if all samples are loaded, else a number specifying how many are loaded.
    clustering_algorithm : "DPGMM"(default) for a Dirichlet Process Gaussian Mixture Model,
                           "HDBSCAN"
    min_cluster_size : minimum acceptable size of a cluster (to not consider clusters of outliers)
    max_clusters_num : maximum number of clusters (required for DPGMM)
    max_exploration_iterations : maximum number of exploration iterations
    background_samples : None(default) if there is no background knowledge
                         background data that account for the background knowledge
    alpha : contrast trade-off hyperparameter between having high original data variance and low background variance
    mu : scaling hyperparameter that accounts for the different scales of PCA(cPCA) and kurtosis terms
    seed: seed selected for reproducibility
    exploration: True for iterative data exploration of a dataset,
                 False for a single 2D data projection that removes
                 the structure of the background data from the structure
                 of the original data

    """

    def __init__(self,
                 model: str,
                 dataset_name: str,
                 input_data,
                 data_labels,
                 clustering_algorithm: str,
                 background_samples,
                 alpha: float,
                 mu: float,
                 **kwargs
                 ):

        self.model = model
        self.dataset_name = dataset_name
        self.input_data = input_data
        self.data_labels = data_labels
        self.clustering_algorithm = clustering_algorithm
        self.background_samples = background_samples
        self.alpha = alpha
        self.mu = mu
        self.kwargs = kwargs


    def run_exploration(self):
        folder_to_write = self.model
        results_dir = os.path.join(os.getcwd(), 'Results')
        dataset_dir = os.path.join(results_dir, self.dataset_name)
        final_dir = os.path.join(dataset_dir, folder_to_write)
        # Create directories if they do not exist
        os.makedirs(final_dir, exist_ok=True)
        if self.background_samples is None:  # assuming no prior knowledge
            data_dict = {"input_data": self.input_data}
            input_data_proc = standardize_data(flag=False, data=data_dict)
            input_data_transf = input_data_proc["input_data"]
            path = final_dir
            if self.model == 'IMAPCE':
                method = "PCA + kPP"
                kwargs = {'root': path,
                          'method': method,
                          'alpha': None,
                          'unexplored_data': None,
                          'mu': self.mu,
                          'iteration': 0,
                          'seed': self.kwargs['seed']}
            else:
                kwargs = {'num_alphas': 40,
                          'max_log_alpha': 3
                          }
            proj = CalculateProjection(model_name=self.model, input_data=input_data_transf, prior_data=None, **kwargs)
            root = path
            plt.clf()
        else:  # starting with some background knowledge associated with background_samples
            unexplored_data = self.input_data
            marked_data = self.background_samples
            data_dict = {"input_data": self.input_data, "background_data": marked_data,
                         "unexplored_data": unexplored_data}
            data_proc = standardize_data(flag=True, data=data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            unexplored_data_transf = data_proc["unexplored_data"]
            root = final_dir + '\\Background_case\\'
            os.makedirs(root, exist_ok=True)
            if self.model == 'IMAPCE':
                method = "cPCA + kPP"
                kwargs = {'root': root,
                          'method': method,
                          'unexplored_data': unexplored_data_transf,
                          'alpha': self.alpha,
                          'mu': self.mu,
                          'iteration': 0,
                          'seed': self.kwargs['seed'], }
            else:
                kwargs = {'num_alphas': 40,
                          'max_log_alpha': 3, }
            proj = CalculateProjection(model_name=self.model, input_data=input_data_transf,
                                       prior_data=marked_data_transf, **kwargs)
            if self.dataset_name == "MNIST and FMNIST" or self.dataset_name == "CIFAR-FMNIST":
                metrics_lin = evaluate_classifier(classifier_type='Linear', embs=proj,
                                                  labels=self.data_labels, seeds=10)
                metrics_svm = evaluate_classifier(classifier_type='SVM', embs=proj,
                                                  labels=self.data_labels, seeds=10)
                print_metrics(metrics=metrics_lin, classifier_type='Linear')
                print_metrics(metrics=metrics_svm, classifier_type='SVM')
            elif self.dataset_name == 'synthetic':
                prior_labels = self.data_labels[:, 1]
                Evaluate_NLS(dataset_name=self.dataset_name, embs=proj,
                             labels=prior_labels, prior_col=None)
            elif self.dataset_name == 'adult':
                _, non_zero_col = np.where(self.background_samples != 0)
                prior_col = np.unique(non_zero_col)
                Evaluate_NLS(dataset_name=self.dataset_name, embs=proj,
                             labels=self.data_labels, prior_col=prior_col)
        plot_embeddings(dataset_name=self.dataset_name, path=root, embs=proj, labels=self.data_labels)
        return

    def Iterative_exploration(self, min_cluster_size, max_clusters_num, max_exploration_iterations):
        folder_to_write = self.model
        results_dir = os.path.join(os.getcwd(), 'Results')
        dataset_dir = os.path.join(results_dir, self.dataset_name)
        final_dir = os.path.join(dataset_dir, folder_to_write)
        # Create directories if they do not exist
        os.makedirs(final_dir, exist_ok=True)
        labels = np.unique(self.data_labels)
        label_mapping = {label: index for index, label in enumerate(labels)}
        gt_labels = np.vectorize(label_mapping.get)(self.data_labels)
        pred_labels = np.zeros(self.data_labels.shape[0])
        non_patterns_ind = np.arange(self.input_data.shape[0])
        total_predictions = []
        if self.background_samples is None:  # assuming no prior knowledge
            bg_patterns_ind = None
            patterns_ind = None
            non_patterns_ind = np.arange(self.input_data.shape[0])
            unexplored_data = self.input_data
            data_dict = {"input_data": self.input_data}
            input_data_proc = standardize_data(flag=False, data=data_dict)
            input_data_transf = input_data_proc["input_data"]
            path = final_dir
            if self.model == 'IMAPCE':
                method = "PCA + kPP"
                kwargs = {'root': path,
                          'method': method,
                          'alpha': None,
                          'unexplored_data': None,
                          'mu': self.mu,
                          'iteration': 0,
                          'seed': self.kwargs['seed']}
            else:
                kwargs = {'num_alphas': 40,
                          'max_log_alpha': 3
                          }
            proj = CalculateProjection(model_name=self.model, input_data=input_data_transf, prior_data=None, **kwargs)
            unexplored_proj = proj
            plt.clf()
        else:
            patterns_ind = common_samples_ind(self.input_data, self.background_samples)
            bg_patterns_ind = patterns_ind
            non_patterns_ind = np.setdiff1d(non_patterns_ind, patterns_ind)
            unexplored_data = self.input_data[non_patterns_ind]
            marked_data = self.background_samples
            data_dict = {"input_data": self.input_data, "background_data": marked_data,
                         "unexplored_data": unexplored_data}
            data_proc = standardize_data(flag=True, data=data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            unexplored_data_transf = data_proc["unexplored_data"]
            root = final_dir + '\\Background_case\\'
            if self.model == 'IMAPCE':
                method = "cPCA + kPP"
                kwargs = {'root': root,
                          'method': method,
                          'unexplored_data': unexplored_data_transf,
                          'alpha': self.alpha,
                          'mu': self.mu,
                          'iteration': 0,
                          'seed': self.kwargs['seed'], }
            else:
                method = 'cPCA'
                kwargs = {'num_alphas': 40,
                          'max_log_alpha': 3, }
            proj = CalculateProjection(model_name=self.model, input_data=input_data_transf,
                                       prior_data=marked_data_transf, **kwargs)
            # proj, proj_matrix = CalculateProjection(model_name=self.model, root=root, method=method,
            #                                         input_data=input_data_transf, prior_data=marked_data_transf,
            #                                         unexplored_data=unexplored_data_transf, alpha=self.alpha,
            #                                         mu=self.mu, iteration=0, seed=self.kwargs['seed'])
        non_patterns_labels = self.data_labels[non_patterns_ind]
        gt_labels_num = np.zeros((labels.shape[0]))
        for i in range(labels.shape[0]):
            class_num = np.squeeze(np.asarray(np.where(self.data_labels == labels[i])))
            gt_labels_num[i] = class_num.shape[0]
            print(f'Number of {labels[i]} instances is: {class_num.shape[0]}')
        for i in range(max_exploration_iterations):
            print(f'The number of unexplored samples are {non_patterns_ind.shape[0]} out of {self.input_data.shape[0]}')
            if unexplored_data.shape[0] > min_cluster_size:  # Continue the exploration process
                predicted_labels, m, cov_mat = clustering(clustering_algorithm=self.clustering_algorithm,
                                                          proj=unexplored_proj,
                                                          min_cluster_size=min_cluster_size,
                                                          max_num_clusters=max_clusters_num)
                mahalanobis_distances, selected_cluster, outliers_flag = mahalanobis_distances_between_clusters(predicted_labels,
                                                                                                                unexplored_proj,
                                                                                                                mean_matrix=m,
                                                                                                                Cluster_cov_matrix=cov_mat,
                                                                                                                min_clust_size=min_cluster_size)
                selected_cluster_val = np.array([selected_cluster])
                clusters_id = np.where(outliers_flag == 0)[0]
                num_clusters = clusters_id.shape[0]
                if num_clusters == 2:
                    selected_cluster_val = np.squeeze(selected_cluster_val)
            if selected_cluster is None or unexplored_data.shape[0] < min_cluster_size:  # The exploration is over
                if bg_patterns_ind is not None:
                    pred_labels = np.delete(pred_labels, bg_patterns_ind)
                    gt_labels = np.delete(gt_labels, bg_patterns_ind)
                print(f"The {unexplored_proj.shape[0]} left and unexplored points are outliers")
                Evaluation_iterative(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels)  # Metrics
                break
            for count, selected_cluster in enumerate(selected_cluster_val):
                selected_cluster = selected_cluster_val[count]
                plot_clustering_results(root=path,
                                        iteration=i, proj=proj, patterns_ind=patterns_ind,
                                        non_patterns_ind=non_patterns_ind, predicted_labels=predicted_labels,
                                        unexplored_data=unexplored_proj, non_patterns_labels=non_patterns_labels,
                                        total_labels=labels, colors=colors)
                if patterns_ind is not None:
                    new_cluster_ind = np.where(predicted_labels == selected_cluster)[0]
                    non_cluster_ind = np.where(predicted_labels != selected_cluster)[0]
                    cluster_points = unexplored_proj[new_cluster_ind]
                    non_cluster_points = unexplored_proj[non_cluster_ind]
                    new_patterns_ind = common_samples_ind(proj, cluster_points)
                    pred_labels[new_patterns_ind] = i + count + 1
                    patterns_ind = np.concatenate((patterns_ind, new_patterns_ind))
                    if non_cluster_points.shape[0] != 0:
                        non_patterns_ind = np.intersect1d(non_patterns_ind,
                                                          common_samples_ind(proj, non_cluster_points))
                        unexplored_data = self.input_data[non_patterns_ind]
                        new_marked_data = self.input_data[new_patterns_ind]
                    marked_data = np.concatenate((marked_data, new_marked_data))
                    new_marked_data_class = self.data_labels[new_patterns_ind]
                else:
                    patterns_ind = np.where(predicted_labels == selected_cluster)[0]
                    new_patterns_ind = patterns_ind
                    non_patterns_ind = np.where(predicted_labels != selected_cluster)[0]
                    pred_labels[new_patterns_ind] = i + count + 1
                    unexplored_data = self.input_data[non_patterns_ind]
                    marked_data = self.input_data[patterns_ind]
                    new_marked_data = marked_data
                    new_marked_data_class = self.data_labels[patterns_ind]
                print(f'Iteration = {i}')
                pred_per_cluster = np.zeros((labels.shape[0]))
                for j in range(np.unique(new_marked_data_class).shape[0]):
                    id = int(np.where(labels == np.unique(new_marked_data_class)[j])[0])
                    pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                    pred_per_cluster[id] = pred_class_num.shape[1]
                    print(f'Number of {np.unique(new_marked_data_class)[j]} instances is: {pred_class_num.shape[1]}')
                total_predictions.append(pred_per_cluster)
                with open(f'{path}\\Proj_{i}_{count}_{colors[selected_cluster]}_cluster.txt', 'w') as f:  # has the classes for the points of the most distinct cluster
                    for j in range(np.unique(new_marked_data_class).shape[0]):
                        pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                        f.write(f'Number of {np.unique(new_marked_data_class)[j]} instances is: {pred_class_num.shape[1]}\n')
            if unexplored_data.shape[0] < min_cluster_size:
                print(f'The {unexplored_data.shape[0]} remaining (unexplored) points are outliers')
                Evaluation_iterative(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels)
                break
            data_dict = {"input_data": self.input_data, "background_data": marked_data,
                         "unexplored_data": unexplored_data}
            data_proc = standardize_data(flag=True, data=data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            unexplored_data_transf = data_proc["unexplored_data"]
            if self.model == 'IMAPCE':
                method = "cPCA + kPP"
                kwargs = {'root': path,
                          'method': method,
                          'unexplored_data': unexplored_data_transf,
                          'alpha': self.alpha,
                          'mu': self.mu,
                          'iteration': i+1,
                          'seed': self.kwargs['seed'], }
            else:
                kwargs = {'num_alphas': 40,
                          'max_log_alpha': 3, }
            proj = CalculateProjection(model_name=self.model, input_data=input_data_transf,
                                       prior_data=marked_data_transf, **kwargs)
            unexplored_proj = proj[non_patterns_ind]
            non_patterns_labels = self.data_labels[non_patterns_ind]
