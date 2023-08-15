import os, random, pandas
import numpy as np
from funcs import load_dataset, standardize_data, CalculateProjection, common_samples_ind, clustering
from funcs import mahalanobis_distances_between_clusters, SelectFromCollection, plot_clustering_results, to_one_hot
from Evaluation import Evaluation, LowDim_classifier, normalized_Laplacian_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

colors=['red','green','blue','yellow','orange','cyan','purple','brown','pink','grey']
plt.rcParams.update({'font.size': 7})
class IMAPCE():
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
    alpha : contrast trade-off hyperparameter between having high original data variance and low background variance
    mu : scaling hyperparameter that accounts for the different scales of PCA(cPCA) and kurtosis terms
    seed: seed selected for reproducibility
    exploration: True for iterative data exploration of a dataset,
                 False for a single 2D data projection that removes
                 the structure of the background data from the structure
                 of the original data

    """

    def __init__(self,
                 dataset_name: str,
                 num_samples: int,
                 selection: str,
                 clustering_algorithm: str,
                 min_cluster_size: int,
                 max_clusters_num : int,
                 max_exploration_iterations: int,
                 background_samples ,
                 alpha,
                 mu ,
                 seed,
                 exploration :bool = True
                 ):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.selection = selection
        self.clustering_algorithm = clustering_algorithm
        self.min_cluster_size = min_cluster_size
        self.max_clusters_num = max_clusters_num
        self.max_exploration_iterations = max_exploration_iterations
        self.exploration = exploration
        self.background_samples = background_samples
        self.alpha = alpha
        self.mu = mu
        self.seed = seed


    def Explore(self):

        folder_to_write = 'IMAPCE'
        if not os.path.exists(os.getcwd()+ '\\Results\\'):
            os.mkdir(os.getcwd()+ '\\Results\\')
        root = os.getcwd() + '\\Results\\'
        if not os.path.exists(root + self.dataset_name):
            os.mkdir(root + self.dataset_name)
        root = os.getcwd() + '\\Results\\' + self.dataset_name + '\\'
        if not os.path.exists(root):
            os.mkdir(root)
        root = root + folder_to_write + '\\'  #replace with the folder you want to save the results
        if not os.path.exists(root):
            os.mkdir(root)
        input_data, labels_data = load_dataset(self.dataset_name, self.num_samples)
        original_data = input_data
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
        if self.background_samples is None: # assuming no prior knowledge
            bg_patterns_ind = None
            patterns_ind = None
            non_patterns_ind = np.arange(input_data.shape[0])
            unexplored_data = input_data
            data_dict = {"input_data" : input_data}
            input_data_proc = standardize_data(flag = False, data = data_dict)
            input_data_transf = input_data_proc["input_data"]
            path = root + 'seed' + str(self.seed)
            if not os.path.exists(path):
                os.mkdir(path)
            path = path + '\\alpha_' + str(self.alpha) + ' mu_' + str(self.mu) + '\\'
            if not os.path.exists(path):
                os.mkdir(path)
            alpha = None
            method = "PCA + kPP"
            proj, proj_matrix = CalculateProjection(root = path, method = method, input_data = input_data_transf,
                                                    background_data = None, unexplored_data = None, alpha = alpha,
                                                    mu = self.mu, iteration = 0, seed = self.seed)
            unexplored_proj = proj
            plt.clf()
        else: # starting with some background knowledge associated with background_samples
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
            unexplored_data_transf = data_proc["unexplored_data"]
            root = root + '\\Background_case\\'
            if not os.path.exists(root):
                os.mkdir(root)
            path = root + 'seed' + str(self.seed) + '\\'
            if not os.path.exists(path):
                os.mkdir(path)
            path = path + '\\alpha_' + str(self.alpha) + ' mu_' + str(self.mu) + '\\'
            if not os.path.exists(path):
                os.mkdir(path)
            method = "cPCA + kPP"
            proj, proj_matrix = CalculateProjection(root = path, method = method, input_data = input_data_transf,
                                                    background_data = marked_data_transf,
                                                    unexplored_data = unexplored_data_transf, alpha = self.alpha,
                                                    mu = self.mu, iteration = 0, seed = self.seed)
            unexplored_proj = proj[non_patterns_ind]
            if self.exploration == False: # Plot the computed data projection
                plt.clf()
                if self.dataset_name == "MNIST and FMNIST":
                    str_labels = np.array(['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
                    for i,l in enumerate(np.unique(labels_data)):
                        id = np.where(labels_data == l)[0]
                        a = plt.scatter(unexplored_proj[id, 0], unexplored_proj[id, 1], marker ='.', color = colors[i], label = str_labels[l])
                    # plt.show()
                    xax = a.axes.get_xaxis()
                    xax.set_visible(False)
                    yax = a.axes.get_yaxis()
                    yax.set_visible(False)
                    plt.tight_layout()
                    plt.legend(loc='best', prop={'size': 20})
                    plt.savefig(path + "Data_Projection.pdf")
                    test_acc_lin = np.zeros((10, 1))
                    test_f1_lin = np.zeros((10, 1))
                    test_acc_svm = np.zeros((10, 1))
                    test_f1_svm = np.zeros((10, 1))
                    for seed in range(10):
                        test_acc_lin[seed], test_f1_lin[seed] = LowDim_classifier('Linear', unexplored_proj, labels_data, seed)
                        test_acc_svm[seed], test_f1_svm[seed] = LowDim_classifier('SVM', unexplored_proj, labels_data, seed)
                    print('Mean test accurary for Linear classifier is: ' + str(np.mean(test_acc_lin)))
                    print('Std of test accurary for Linear classifier is: ' + str(np.std(test_acc_lin)))
                    print('Mean test accurary for SVM classifier is: ' + str(np.mean(test_acc_svm)))
                    print('Std of test accurary for SVM classifier is: ' + str(np.std(test_acc_svm)))
                    return
                elif self.dataset_name == 'Single Cell':
                    str_labels = np.array(['pre 1', 'post 1'])
                elif self.dataset_name == 'synthetic':
                    data_df = pandas.read_csv("synthetic_dataset.csv")
                    data_df.head()
                    label_col = "batch"
                    L = np.asarray(data_df[label_col])
                    marker = np.array(['o', '^', 's'])
                    for i,l in enumerate(np.unique(labels_data)):
                        id = np.where(labels_data == l)[0]
                        for j, k in enumerate(np.unique(L)):
                            id2 = np.where(L == k)[0]
                            idt = np.intersect1d(id, id2)
                            a = plt.scatter(unexplored_proj[idt, 0], unexplored_proj[idt, 1], marker =marker[j], color = colors[i], facecolors = 'None', alpha = 0.3)
                    labels_onehot = to_one_hot(np.expand_dims(L, 1))
                    labels = ["Cluster 1 of dims 5-6", "Cluster 2 of dims 5-6", "Cluster 3 of dims 5-6"]
                    f = lambda m,c,fill: plt.plot([],[], marker=m, color=c, ls="none", fillstyle = fill)[0]
                    handles = [f("s", colors[i], 'full') for i in range(3)]
                    handles += [f(marker[i], "k", 'none') for i in range(2)]
                    labels += ["Cluster 1 of dims 1-4", "Cluster 2 of dims 1-4"]
                    plt.legend(handles, labels, loc='best', framealpha=1)
                    xax = a.axes.get_xaxis()
                    xax.set_visible(False)
                    yax = a.axes.get_yaxis()
                    yax.set_visible(False)
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(path + "IMAPCE_synthetic.pdf")
                    # LowDim_classifier(unexplored_proj, labels_data, self.seed)
                    NLS = np.zeros((10, 1))
                    counter = 0
                    for neighbors in range(10, 110, 10):
                        NLS[counter] = normalized_Laplacian_score(unexplored_proj, neighbors, labels_onehot)
                        print("Normalized Laplacian score for k = " + str(neighbors) + " is " + str(NLS[counter]))
                        counter+=1
                    with open('NLS_synthetic_IMACE.npy', 'wb') as f:
                        np.save(f, NLS)
                    return
                elif self.dataset_name == 'synthetic ct-SNE':
                    labels_onehot = np.zeros((labels_data.shape[0], np.unique(labels_data).shape[0]))
                    for i,l in enumerate(np.unique(labels_data)):
                        id = np.where(labels_data == l)[0]
                        labels_onehot[id, i] = 1
                        plt.scatter(unexplored_proj[id, 0], unexplored_proj[id, 1], marker ='.', color = colors[i])
                    # plt.show()
                    plt.savefig(path + "Data_Projection.pdf")
                    # LowDim_classifier(unexplored_proj, labels_data, self.seed)
                    for neighbors in range(10, 110, 10):
                        NLS = normalized_Laplacian_score(unexplored_proj, neighbors, labels_onehot)
                        print("Normalized Laplacian score for k = " + str(neighbors) + " is " + str(NLS))
                    return
                elif self.dataset_name == 'adult':
                    markers = ['o', '^']
                    marker_size = 20
                    gender_col = labels_data[:, 0]
                    gender_onehot = to_one_hot(np.expand_dims(gender_col, axis =1 ))
                    ethnicity_col = labels_data[:, 1]
                    ethnicity_onehot= to_one_hot(np.expand_dims(ethnicity_col, axis =1 ))
                    gender_ethnicity_onehot = to_one_hot(labels_data[:, 0:2])
                    income_col = labels_data[:, 2]
                    for c in np.unique(gender_col):
                        for m in np.unique(ethnicity_col):
                            for f in np.unique(income_col):
                                mask = (gender_col == c) & (ethnicity_col == m) & (income_col == f)
                                facecolors = colors[c] if f else 'none'
                                edgecolors = 'none' if f else colors[c]
                                marker_size = 12 if m == 1 else marker_size
                                a = plt.scatter(unexplored_proj[mask, 0], unexplored_proj[mask, 1], marker_size, edgecolors=edgecolors,
                                            marker=markers[m], facecolors=facecolors, alpha=0.8, linewidth=0.6)
                    labels = ["female", "male", "white", "other"]
                    fill = ['full', 'none']
                    f = lambda m,c,fill: plt.plot([],[], marker=m, color=c, ls="none", fillstyle = fill)[0]
                    handles = [f("s", colors[i], 'full') for i in range(2)]
                    handles += [f(markers[i], "k", 'none') for i in range(2)]
                    handles += [f("s", "k", fill[i]) for i in range(2)]
                    labels += ["income<50k", "income>50k"]
                    plt.legend(handles, labels, loc='best', framealpha=1)
                    xax = a.axes.get_xaxis()
                    xax.set_visible(False)
                    yax = a.axes.get_yaxis()
                    yax.set_visible(False)
                    plt.tight_layout()
                    plt.savefig(path + 'uci_Adult.pdf')
                    NLS_gender_ethnicity_IMAPCE = np.zeros((10, 1))
                    NLS_gender_IMAPCE = np.zeros((10, 1))
                    NLS_ethnicity_IMAPCE = np.zeros((10, 1))
                    counter = 0
                    for neighbors in range(10, 110, 10):
                        NLS_gender_ethnicity_IMAPCE[counter] = normalized_Laplacian_score(unexplored_proj, neighbors, gender_ethnicity_onehot)
                        NLS_gender_IMAPCE[counter] = normalized_Laplacian_score(unexplored_proj, neighbors, gender_onehot)
                        NLS_ethnicity_IMAPCE[counter] = normalized_Laplacian_score(unexplored_proj, neighbors, ethnicity_onehot)
                        print(str(NLS_ethnicity_IMAPCE[counter]))
                        counter+=1
                    # print("Normalized Laplacian score for k = " + str(neighbors) + " is " + str(NLS_gender_ethnicity_cPCA[counter]))
                    with open('NLS_adult_IMAPCE_gender_ethnicity_prior_gender_labels.npy', 'wb') as f:
                        np.save(f, NLS_gender_IMAPCE)
                    with open('NLS_adult_IMAPCE_gender_ethnicity_prior_ethnicity_labels.npy', 'wb') as f:
                        np.save(f, NLS_ethnicity_IMAPCE)
                    with open('NLS_adult_IMAPCE_gender_ethnicity_prior_gender_ethnicity_labels.npy', 'wb') as f:
                        np.save(f, NLS_gender_ethnicity_IMAPCE)
                    # for neighbors in range(10, 110, 10):
                    #     NLS = normalized_Laplacian_score(unexplored_proj, neighbors, random_gender_ethnicity_onehot)
                    #     print("Normalized Laplacian score for k = " + str(neighbors) + " is " + str(NLS))
                    return
        non_patterns_labels = labels_data[non_patterns_ind]
        if self.dataset_name != 'MNIST and FMINST' and self.dataset_name != 'Odour data':
            gt_labels_num = np.zeros((labels.shape[0]))
            for i in range(np.unique(labels_data).shape[0]):
                class_num = np.squeeze(np.asarray(np.where(labels_data == np.unique(labels_data)[i])))
                gt_labels_num[i] = class_num.shape[0]
                print('Number of ' + str(np.unique(labels_data)[i]) + ' instances is: ' + str(class_num.shape[0]))
        for i in range(self.max_exploration_iterations):
            print('The number of unexplored samples are: ' + str(non_patterns_ind.shape[0]) + ' out of ' + str(input_data.shape[0]) + ' data samples')
            if self.selection == "automatic": # automatic selection of the most distinct cluster
                if unexplored_data.shape[0] > self.min_cluster_size: # There are more data to be explored than outliers
                    predicted_labels, m, cov_mat = clustering(clustering_algorithm = self.clustering_algorithm,
                                                              proj = unexplored_proj,
                                                              min_cluster_size = self.min_cluster_size,
                                                              max_num_clusters = self.max_clusters_num)
                    mahalanobis_distances, selected_cluster, outliers_flag = mahalanobis_distances_between_clusters(predicted_labels,
                                                                                                                    unexplored_proj,
                                                                                                                    mean_matrix = m,
                                                                                                                    Cluster_cov_matrix = cov_mat,
                                                                                                                    min_clust_size = self.min_cluster_size)
                    selected_cluster_val = np.array([selected_cluster])
                    clusters_id = np.where(outliers_flag == 0)[0]
                    num_clusters = clusters_id.shape[0]
                    if num_clusters == 2:
                        selected_cluster_val = np.squeeze(selected_cluster_val)
                if selected_cluster is None or unexplored_data.shape[0] < self.min_cluster_size: # The exploration is over
                    if bg_patterns_ind is not None:
                        pred_labels = np.delete(pred_labels, bg_patterns_ind)
                        gt_labels = np.delete(gt_labels, bg_patterns_ind)
                    print("The " + str(unexplored_proj.shape[0]) + " left and unexplored points are outliers")
                    JS, MJS, H, C, V = Evaluation(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels) # Evaluation of the results
                    break
                for count,selected_cluster in enumerate(selected_cluster_val):
                    selected_cluster = selected_cluster_val[count]
                    plot_clustering_results(root = path, multilabel=False, most_informative_cluster = selected_cluster, iteration = i,
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
                        if non_cluster_points.shape[0] !=   0:
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
                    with open(path + 'Proj' + str(i) + '_'+ str(count) + '_' + str(colors[selected_cluster]) + '_cluster'  +  '.txt', 'w') as f: # has the classes for the points of the most distinct cluster
                        for j in range(np.unique(new_marked_data_class).shape[0]):
                            pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                            f.write('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                            f.write('\n')
                    if self.exploration == False:
                        return
            else: # selection is semi-automatic
                count = 0
                while(True):
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
                    plt.savefig(path + 'Proj' + str(i) + '_' + str(count) + '.pdf')
                    pred_per_cluster = np.zeros((labels.shape[0]))
                    for j in range(np.unique(new_marked_data_class).shape[0]):
                        id = int(np.where(labels == np.unique(new_marked_data_class)[j])[0])
                        pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                        pred_per_cluster[id] = pred_class_num.shape[1]
                        print('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                    total_predictions.append(pred_per_cluster)
                    with open(path + 'Instances_per_cluster' + str(i) + '_'+ str(count) + '.txt', 'w') as f:
                        for j in range(np.unique(new_marked_data_class).shape[0]):
                            pred_class_num = np.asarray(np.where(new_marked_data_class == np.unique(new_marked_data_class)[j]))
                            f.write('Number of ' + str(np.unique(new_marked_data_class)[j]) + ' instances is: ' + str(pred_class_num.shape[1]))
                            f.write('\n')
                    count += 1
            if unexplored_data.shape[0] < self.min_cluster_size:
                print("The "+ str(unexplored_data.shape[0])  +" left and unexplored points are outliers")
                JS, MJS, H, C, V = Evaluation(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels)
                break
            data_dict = {"input_data" : input_data, "background_data" : marked_data, "unexplored_data" : unexplored_data}
            data_proc = standardize_data(flag = True, data = data_dict)
            input_data_transf = data_proc["input_data"]
            marked_data_transf = data_proc["background_data"]
            unexplored_data_transf = data_proc["unexplored_data"]
            method = "cPCA + kPP"
            alpha = self.alpha
            proj, proj_matrix = CalculateProjection(root=path, method = method, input_data = input_data_transf,
                                                    background_data = marked_data_transf,
                                                    unexplored_data = unexplored_data_transf, alpha = alpha,
                                                    mu = self.mu, iteration = i + 1, seed = self.seed)
            unexplored_proj = proj[non_patterns_ind]
            non_patterns_labels = labels_data[non_patterns_ind]