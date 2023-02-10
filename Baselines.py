from contrastive import CPCA
from funcs import load_dataset, standardize_data, common_samples_ind, clustering, sum_, diag_
from funcs import plot_clustering_results, mahalanobis_distances_between_clusters, SelectFromCollection
from Evaluation import Evaluation
import matplotlib.pyplot as plt
import numpy as np
import time, os
from scipy.stats import  kurtosis
from sklearn.decomposition import PCA

colors=['red','green','blue','yellow','orange','cyan','purple','brown','pink','grey']

def find_alphas( A, B, non_patterns_ind, alphas):
    """
    Find alphas based on the kurtosis value of the cPCA projections considering the unexplored samples

    :param A: foreground data passed in cPCA
    :param B: background data passed in cPCA
    :param non_patterns_ind: indices of data samples which remain unexplored and whose kurtosis is measured
    :param alphas: the vector of alphas which are used to compute cPCA data projections

    """

    K = np.zeros((alphas.shape[0], 2))
    K_sum = np.zeros((alphas.shape[0], 1))
    total_time = 0
    for i, a in enumerate(alphas):
        t0 = time.time()
        mdl = CPCA()
        proj = mdl.fit_transform(A, B, alpha_selection='manual', alpha_value= a)
        K[i, :] = kurtosis(proj[non_patterns_ind])
        K_sum[i] = K[i, 0] + K[i, 1]
        t1 = time.time() - t0
        total_time += t1
        print('Convergence for alpha = ' + str(np.round(a, decimals = 2)) + ' took ' + str(np.round(t1, decimals = 2)) + ' seconds, alphas checked : ' + str(i+1) + '/' + str(alphas.shape[0]) )
    Uni_K_sorted_ind =np.argsort(K_sum, axis = 0)
    alphas_sorted_K = alphas[Uni_K_sorted_ind]
    print('Total time for tuning alpha is : ' + str(np.round(total_time, decimals = 2)) + ' seconds')
    print('Selected alpha is: ' + str(np.round(alphas_sorted_K[0], decimals = 2)))
    return alphas_sorted_K

class Baselines():
    """
    Informative MAnifold Projections for Cluster Exploration

    Parameters:
    --------
    dataset_name : string of the dataset's name
    num_samples : the number of samples of the selected dataset,
                  "All" if all samples are loaded, else a number specifying how many are loaded.
    selection : "automatic"(default) if the most distinct cluster is computed at each data projection,
                "semi-automatic" if the user manually wants to select the clusters at each data projection.
    clustering_algorithm : "DPGMM"(default) for a Dirichlet Process Gaussian Mixture Model,
                           "HDBSCAN"
    min_cluster_size : minimum acceptable size of a cluster (to not consider clusters of outliers)
    max_clusters_num : maximum number of clusters (required for DPGMM)
    max_exploration_iterations : maximum number of exploration iterations
    background_samples : None(default) if there is no background knowledge
                         background data that account for the background knowledge
    exploration: True for iterative data exploration of a dataset,
                 False for a single 2D data projection that removes
                 the structure of the background data from the structure
                 of the original data

    """
    def __init__(self,
                 dataset_name: str,
                 num_samples: int,
                 method: str,
                 num_alphas :int,
                 selection: str,
                 clustering_algorithm: str,
                 min_cluster_size: int,
                 max_clusters_num : int,
                 max_exploration_iterations: int,
                 background_samples ,
                 exploration :bool = True
                 ):

        self.dataset_name = dataset_name
        self.method = method
        self.num_samples = num_samples
        self.num_alphas = num_alphas
        self.selection = selection
        self.clustering_algorithm = clustering_algorithm
        self.min_cluster_size = min_cluster_size
        self.max_clusters_num = max_clusters_num
        self.max_exploration_iterations = max_exploration_iterations
        self.exploration = exploration
        self.background_samples = background_samples


    def Explore(self):
        folder_to_write = self.method
        root = os.getcwd() + '\\Results\\' + self.dataset_name + '\\' + folder_to_write + '\\'
        if not os.path.exists(root):
            os.mkdir(root)
        input_data, labels_data = load_dataset(self.dataset_name, self.num_samples)
        labels = np.unique(labels_data)
        gt_labels = np.zeros((labels_data.shape[0]))
        for i,l in enumerate(np.unique(labels_data)):
            label_ind = np.where(labels_data == l)[0]
            gt_labels[label_ind] = i
        if self.selection == "semi-automatic":
            root = root + 'SA\\'
        pred_labels = np.zeros(labels_data.shape[0])
        non_patterns_ind = np.arange(input_data.shape[0])
        total_predictions = []
        if self.background_samples is None:
            bg_patterns_ind = None
            patterns_ind = None
            non_patterns_ind = np.arange(input_data.shape[0])
            unexplored_data = input_data
            data_dict = {"input_data" : input_data}
            input_data_proc = standardize_data(flag = False, data = data_dict)
            input_data_transf = input_data_proc["input_data"]
            pca = PCA(n_components = 2)
            pca.fit(input_data_transf) # input has shape n x d
            Theta = np.transpose(pca.components_) # finding the two first principal components, shape : d x 2
            proj = input_data_transf@Theta # projecting the data on the two first principal components
            unexplored_proj = proj
        else:
            if self.exploration == False:
                non_patterns_ind = np.arange(input_data.shape[0])
                unexplored_data = input_data
                patterns_ind = None
                marked_data = self.background_samples
            else:
                patterns_ind = common_samples_ind(input_data, self.background_samples)
                bg_patterns_ind= patterns_ind
                non_patterns_ind = np.setdiff1d(non_patterns_ind, patterns_ind)
                unexplored_data = input_data[non_patterns_ind]
                marked_data = self.background_samples
            data_dict = {"input_data" : input_data, "background_data" : marked_data, "unexplored_data" : unexplored_data}
            data_proc = standardize_data(flag = True, data = data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            root = root + 'Background_case\\'
            if not os.path.exists(root):
                os.mkdir(root)
            if self.selection == "semi-automatic":
                root = root + '\\SA\\'
            mdl = CPCA()
            if self.method == 'original cPCA':
                proj = mdl.fit_transform(input_data_transf, marked_data_transf, n_alphas=1,  max_log_alpha=3, n_alphas_to_return=1)
                proj = np.squeeze(np.asarray(proj))
            else: # method = Boosted cPCA
                alphas = np.logspace(start = -1, stop = 3, num = self.num_alphas)
                alphas_sorted_K = find_alphas(alphas.shape[0], input_data_transf, marked_data_transf, non_patterns_ind, alphas)
                alpha = alphas_sorted_K[0]
                proj = mdl.fit_transform(input_data_transf, marked_data_transf, alpha_selection='manual', alpha_value = alpha)

            unexplored_proj = proj[non_patterns_ind]
            if self.exploration == False:
                plt.clf()
                str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
                for i,l in enumerate(labels):
                    id = np.where(labels_data == l)[0]
                    if i == 0:
                        opacity = 1
                    else:
                        opacity = 0.5
                    a = plt.scatter(unexplored_proj[id, 0], unexplored_proj[id, 1], marker ='.', color = colors[i], alpha = opacity, label= str_labels[labels[i]])
                xax = a.axes.get_xaxis()
                xax.set_visible(False)
                yax = a.axes.get_yaxis()
                yax.set_visible(False)
                plt.tight_layout()
                plt.legend(loc=4, prop={'size': 20})
                plt.savefig(root + "Data_Projection.pdf")
                return
        non_patterns_labels = labels_data[non_patterns_ind]
        if self.dataset_name != 'MNIST and FMINST':
            gt_labels_num = np.zeros((labels.shape[0]))
            for i in range(np.unique(labels_data).shape[0]):
                class_num = np.squeeze(np.asarray(np.where(labels_data == np.unique(labels_data)[i])))
                gt_labels_num[i] = class_num.shape[0]
                print('Number of ' + str(np.unique(labels_data)[i]) + ' instances is: ' + str(class_num.shape[0]))
        for i in range(self.max_exploration_iterations):
            print('The number of unexplored samples are: ' + str(non_patterns_ind.shape[0]) + ' out of ' + str(input_data.shape[0]) + ' data samples')
            if self.selection == "automatic":
                if unexplored_data.shape[0] > self.min_cluster_size:
                    predicted_labels, m, cov_mat = clustering(clustering_algorithm = self.clustering_algorithm,
                                                              proj = unexplored_proj,
                                                              min_cluster_size = self.min_cluster_size,
                                                              max_num_clusters = self.max_clusters_num)
                    mahalanobis_distances, selected_cluster, outliers_flag = mahalanobis_distances_between_clusters(labeled_points = predicted_labels,
                                                                                                                    proj = unexplored_proj,
                                                                                                                    mean_matrix = m,
                                                                                                                    Cluster_cov_matrix = cov_mat,
                                                                                                                    min_clust_size = self.min_cluster_size)
                    selected_cluster_val = np.array([selected_cluster])
                    clusters_id = np.where(outliers_flag == 0)[0]
                    num_clusters = clusters_id.shape[0]
                    if num_clusters == 2:
                        selected_cluster_val = np.squeeze(selected_cluster_val)
                if selected_cluster is None or unexplored_data.shape[0] < self.min_cluster_size:
                    if bg_patterns_ind is not None:
                        pred_labels = np.delete(pred_labels, bg_patterns_ind)
                        gt_labels = np.delete(gt_labels, bg_patterns_ind)
                    print("The " + str(unexplored_proj.shape[0]) + " left and unexplored points are outliers")
                    JS, MJS, H, C, V = Evaluation(root, total_predictions, labels, gt_labels, gt_labels_num, pred_labels)
                    break
                for count,selected_cluster in enumerate(selected_cluster_val):
                    selected_cluster = selected_cluster_val[count]
                    plot_clustering_results(root = root, most_informative_cluster = selected_cluster, iteration = i,
                                            proj = proj, patterns_ind = patterns_ind, non_patterns_ind = non_patterns_ind,
                                            predicted_labels = predicted_labels, unexplored_data = unexplored_proj,
                                            non_patterns_labels = non_patterns_labels, total_labels = labels, colors = colors)
                    if patterns_ind is not None:
                        new_cluster_ind = np.where(predicted_labels == selected_cluster)[0]
                        non_cluster_ind = np.where(predicted_labels != selected_cluster)[0]
                        cluster_points = unexplored_proj[new_cluster_ind] # has the
                        non_cluster_points = unexplored_proj[non_cluster_ind]
                        new_patterns_ind = common_samples_ind(proj, cluster_points)
                        pred_labels[new_patterns_ind] = i + count + 1
                        patterns_ind = np.concatenate((patterns_ind, new_patterns_ind))
                        if non_cluster_points.shape[0] != 0:
                            non_patterns_ind = np.intersect1d(non_patterns_ind,
                                                              common_samples_ind(proj, non_cluster_points))
                            unexplored_data = input_data[non_patterns_ind]
                            new_marked_data = input_data[new_patterns_ind]
                        marked_data = np.concatenate((marked_data, new_marked_data))
                        new_marked_data_class = labels_data[new_patterns_ind]
                    else:
                        patterns_ind = np.where(predicted_labels == selected_cluster)[0]
                        new_patterns_ind = patterns_ind
                        non_patterns_ind = np.where(predicted_labels != selected_cluster)[0]
                        pred_labels[new_patterns_ind] = i + count + 1
                        unexplored_data = input_data[non_patterns_ind]
                        marked_data = input_data[patterns_ind]
                        new_marked_data = marked_data
                        new_marked_data_class = labels_data[patterns_ind]
                    print('Iteration = ' + str(i))
                    pred_per_cluster = np.zeros((labels.shape[0]))
                    for j in range(np.unique(new_marked_data_class).shape[0]):
                        id = int(np.where(labels == np.unique(new_marked_data_class)[j])[0])
                        pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                        pred_per_cluster[id] = pred_class_num.shape[1]
                        print('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                    total_predictions.append(pred_per_cluster)
                    with open(root + 'Instances_per_cluster' + str(i) + '_'+ str(count) +  '.txt', 'w') as f:
                        for j in range(np.unique(new_marked_data_class).shape[0]):
                            pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                            f.write('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                            f.write('\n')
                    if self.exploration == False:
                        return
            else:
                count = 0
                while(True):# selection is semi-automatic
                    subplot_kw = dict(autoscale_on=True)
                    fig, ax = plt.subplots(subplot_kw=subplot_kw)

                    pts = ax.scatter(unexplored_proj[:, 0], unexplored_proj[:, 1], marker ='.')
                    selector = SelectFromCollection(ax, pts)
                    def accept(event):
                        if event.key == "enter":
                            selector.disconnect()
                            ax.set_title("")
                            fig.canvas.draw()

                    fig.canvas.mpl_connect("key_press_event", accept)
                    ax.set_title("Press enter to accept selected points.")
                    plt.show()
                    if selector.ind == []:
                        break
                    if patterns_ind is not None:
                        n = np.arange(unexplored_proj.shape[0])
                        new_cluster_ind = selector.ind
                        non_cluster_ind = np.setdiff1d(n, selector.ind)
                        cluster_points = unexplored_proj[new_cluster_ind] # has the
                        non_cluster_points = unexplored_proj[non_cluster_ind]
                        new_patterns_ind = common_samples_ind(proj, cluster_points)
                        pred_labels[new_patterns_ind] = i + count + 1
                        patterns_ind = np.concatenate((patterns_ind, new_patterns_ind))
                        non_patterns_ind = np.intersect1d(non_patterns_ind,
                                                          common_samples_ind(proj, non_cluster_points))
                        unexplored_data = input_data[non_patterns_ind]
                        new_marked_data = input_data[new_patterns_ind]
                        marked_data = np.concatenate((marked_data, new_marked_data))
                        new_marked_data_class = labels_data[new_patterns_ind]
                    else:
                        n = np.arange(unexplored_proj.shape[0])
                        patterns_ind = selector.ind
                        new_patterns_ind = patterns_ind
                        pred_labels[new_patterns_ind] = i + count + 1
                        non_patterns_ind = np.setdiff1d(n, selector.ind)
                        unexplored_data = input_data[non_patterns_ind]
                        marked_data = input_data[patterns_ind]
                        new_marked_data = marked_data
                        new_marked_data_class = labels_data[patterns_ind]
                    plt.scatter(proj[non_patterns_ind, 0], proj[non_patterns_ind, 1], marker = '.', color = 'black')
                    plt.scatter(proj[new_patterns_ind, 0], proj[new_patterns_ind, 1], marker = '.', color = 'orange')
                    plt.savefig(root + 'Proj' + str(i) + '_' + str(count) + '.pdf')
                    pred_per_cluster = np.zeros((labels.shape[0]))
                    for j in range(np.unique(new_marked_data_class).shape[0]):
                        id = int(np.where(labels == np.unique(new_marked_data_class)[j])[0])
                        pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                        pred_per_cluster[id] = pred_class_num.shape[1]
                        print('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                    total_predictions.append(pred_per_cluster)
                    with open(root + 'Instances_per_cluster' + str(i) + '_'+ str(count) + '.txt', 'w') as f:
                        for j in range(np.unique(new_marked_data_class).shape[0]):
                            pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                            f.write('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                            f.write('\n')
                    count += 1
            if unexplored_data.shape[0] < self.min_cluster_size:
                print("The "+ str(unexplored_data.shape[0])  +" left and unexplored points are outliers")
                JS, MJS, H, C, V = Evaluation(root, total_predictions, labels, gt_labels, gt_labels_num, pred_labels)
                break
            data_dict = {"input_data" : input_data, "background_data" : marked_data, "unexplored_data" : unexplored_data}
            data_proc = standardize_data(flag = True, data = data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            mdl = CPCA()
            if self.method == 'original cPCA':
                proj = mdl.fit_transform(input_data_transf, marked_data_transf, n_alphas = 1,  max_log_alpha = 3, n_alphas_to_return = 1)
                proj = np.squeeze(np.asarray(proj))
            else: # method = Boosted cPCA
                alphas = np.logspace(start = -1, stop = 3, num = self.num_alphas)
                alphas_sorted_K = find_alphas(input_data_transf, marked_data_transf, non_patterns_ind, alphas)
                alpha = alphas_sorted_K[0]
                proj = mdl.fit_transform(input_data_transf, marked_data_transf, alpha_selection = 'manual', alpha_value = alpha)
            unexplored_proj = proj[non_patterns_ind]
            non_patterns_labels = labels_data[non_patterns_ind]
