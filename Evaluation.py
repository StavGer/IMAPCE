import numpy as np
from sklearn import metrics

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
        f.write('Homogeneity score is : ' + str(metrics.homogeneity_score(gt_labels, pred_labels))
                + ', Completeness score is : ' + str(metrics.completeness_score(gt_labels, pred_labels))
                + ', Harmonic V measure is : ' + str(metrics.v_measure_score(gt_labels, pred_labels)))
    return jaccard_scores, mean_jaccard_score, metrics.homogeneity_score(gt_labels, pred_labels), \
           metrics.completeness_score(gt_labels, pred_labels), metrics.v_measure_score(gt_labels, pred_labels)