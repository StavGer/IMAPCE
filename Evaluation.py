import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import kneighbors_graph

def Evaluation(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels):
    jaccard_scores = np.zeros((labels.shape[0]))
    total_predictions_arr = np.array(total_predictions)
    cluster_topclass_id = np.argmax(total_predictions_arr, axis = 0)
    mean_jaccard_score = 0
    with open(path + 'Evaluation.txt', 'w') as f:
        for j in range(labels.shape[0]):
            cluster_id = cluster_topclass_id[j]
            jaccard_scores[j] = total_predictions_arr[cluster_id, j]/max(gt_labels_num[j], np.sum(total_predictions_arr[cluster_id, :]))
            if jaccard_scores[j] != 0:
                mean_jaccard_score += jaccard_scores[j]*gt_labels_num[j]
            else:
                gt_labels_num[j] = 0
            f.write('Jaccard index score for ' + str(labels[j]) + ' class is: ' + str(jaccard_scores[j]))
            f.write('\n')
        mean_jaccard_score = mean_jaccard_score/np.sum(gt_labels_num)
        f.write('Mean Jaccard score is: ' + str(mean_jaccard_score))
        f.write('\n')
        f.write('NMI is : ' + str(metrics.v_measure_score(gt_labels, pred_labels)))
    return jaccard_scores, mean_jaccard_score, metrics.homogeneity_score(gt_labels, pred_labels), \
           metrics.completeness_score(gt_labels, pred_labels), metrics.v_measure_score(gt_labels, pred_labels)


def LowDim_classifier(method, X, gt_labels, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, gt_labels, test_size=0.25, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if method == 'Linear':
        clf = SGDClassifier()
    elif method =='SVM':
        clf = svm.SVC(kernel='rbf')
    # fit (train) the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average = 'weighted')

def normalized_Laplacian_score(proj, n_neighbors, prior_onehot):
    A = kneighbors_graph(proj, n_neighbors)
    A = A.toarray()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    Dneg = D**(-0.5)
    Dneg[np.isinf(Dneg)] = 0
    Lnorm = Dneg@L@Dneg
    NLS = 0
    for i in range(prior_onehot.shape[1]):
        n_ones = prior_onehot[:, i].T@prior_onehot[:, i]
        NLS += n_ones * prior_onehot[:, i].T@Lnorm@prior_onehot[:, i]/(prior_onehot.shape[0]*n_ones)
    return NLS


def Laplacian_scores(dataset_name, labels):
    if dataset_name == 'synthetic':
        with open('NLS_synthetic_IMACE.npy', 'rb') as f:
            NLS_IMAPCE = np.load(f)
        with open('NLS_synthetic_ctsne.npy', 'rb') as f:
            NLS_ctsne = np.load(f)
        with open('NLS_synthetic_cPCA.npy', 'rb') as f:
            NLS_cPCA = np.load(f)
    elif dataset_name == 'adult':
        if labels == 'ethnicity':
            with open('NLS_adult_IMAPCE_ethnicity_prior_ethnicity_labels.npy', 'rb') as f:
                NLS_IMAPCE = np.load(f)
            with open('NLS_adult_ctsne_ethnicity_prior_ethnicity_labels.npy', 'rb') as f:
                NLS_ctsne = np.load(f)
            with open('NLS_adult_cPCA_ethnicity_prior_ethnicity_labels.npy', 'rb') as f:
                NLS_cPCA = np.load(f)
        elif labels == 'gender':
            with open('NLS_adult_IMAPCE_gender_prior_gender_labels.npy', 'rb') as f:
                NLS_IMAPCE = np.load(f)
            with open('NLS_adult_ctsne_gender_prior_gender_labels.npy', 'rb') as f:
                NLS_ctsne = np.load(f)
            with open('NLS_adult_cPCA_gender_prior_gender_labels.npy', 'rb') as f:
                NLS_cPCA = np.load(f)
        elif labels == 'gender-ethnicity':
            with open('NLS_adult_IMAPCE_gender_ethnicity_prior_gender_ethnicity_labels.npy', 'rb') as f:
                NLS_IMAPCE = np.load(f)
            with open('NLS_adult_ctsne_gender_ethnicity_prior_gender_ethnicity_labels.npy', 'rb') as f:
                NLS_ctsne = np.load(f)
            with open('NLS_adult_cPCA_gender_ethnicity_prior_gender_ethnicity_labels.npy', 'rb') as f:
                NLS_cPCA = np.load(f)
    k = np.array(range(10,110, 10))
    plt.plot(k, NLS_IMAPCE, color = 'blue', marker = '*', label = 'IMAPCE with ' + labels + ' as priors')
    plt.plot(k, NLS_ctsne, color = 'green', marker = 'o', label = 'ctsne with ' + labels + ' as priors')
    plt.plot(k, NLS_cPCA, color = 'yellow', marker = '*', label = 'cPCA with ' + labels + ' as priors')
    plt.legend()
    plt.xlabel("Number of neighbors")
    plt.ylabel("Laplacian score")
    plt.savefig("Laplacian_scores.pdf")
    plt.show()