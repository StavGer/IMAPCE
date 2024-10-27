import numpy as np
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import kneighbors_graph

from funcs import to_one_hot

def Evaluation_iterative(path, total_predictions, labels, gt_labels, gt_labels_num, pred_labels):
    jaccard_scores = np.zeros((labels.shape[0]))
    total_predictions_arr = np.array(total_predictions)
    cluster_topclass_id = np.argmax(total_predictions_arr, axis = 0)
    mean_jaccard_score = 0
    with open(path + 'Evaluation.txt', 'w') as f:
        for j in range(labels.shape[0]):
            cluster_id = cluster_topclass_id[j]
            jaccard_scores[j] = total_predictions_arr[cluster_id, j]/max(gt_labels_num[j],
                                                                         np.sum(total_predictions_arr[cluster_id, :]))
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
    return


def LowDim_classifier(method, X, gt_labels, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, gt_labels, test_size=0.25, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if method == 'Linear':
        clf = SGDClassifier()
    elif method == 'SVM':
        clf = svm.SVC(kernel='rbf')
    # fit (train) the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='weighted'),\
           metrics.recall_score(y_test, y_pred, pos_label=y_test[0])

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


def evaluate_classifier(classifier_type, embs, labels, seeds=10):
    metrics = {
        'accuracy': np.zeros(seeds),
        'f1': np.zeros(seeds),
        'recall': np.zeros(seeds)
    }

    for seed in range(seeds):
        metrics['accuracy'][seed], metrics['f1'][seed], metrics['recall'][seed] = LowDim_classifier(classifier_type, embs, labels, seed)

    return metrics

def Evaluate_NLS(dataset_name, embs, labels, prior_col):
    if dataset_name == 'adult':
        if prior_col == 3:
            prior = 'gender'
            gender_col = labels[:, 0]
            prior_onehot = to_one_hot(np.expand_dims(gender_col, axis=1))
        elif prior_col == 2:
            prior = 'ethnicity'
            ethnicity_col = labels[:, 1]
            prior_onehot= to_one_hot(np.expand_dims(ethnicity_col, axis=1))
        else: # prior is gender-ethnicity
            prior = 'gender-ethnicity'
            prior_onehot = to_one_hot(labels[:, 0:2])
    else:  # synthetic dataset
        prior = 'features of dims 1-4'
        prior_onehot = to_one_hot(np.expand_dims(labels, 1))
    NLS = np.zeros((10, 1))
    for i, neighbors in enumerate(range(10, 110, 10)):
        NLS[i] = normalized_Laplacian_score(embs, neighbors, prior_onehot)
    print(f'Mean NLS for {prior} as prior is: {np.mean(NLS)}')
