import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score, confusion_matrix


def calc_auroc_score(test_score, ood_score):
    total_scores = np.concatenate((test_score,ood_score))
    y_true = np.concatenate((np.zeros_like(test_score),np.ones_like(ood_score)))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, total_scores, pos_label=1)
    auroc_score = metrics.roc_auc_score(y_true, total_scores)
    return auroc_score

def calc_all_scores(test_score, ood_score):
    y_true = np.concatenate((np.zeros_like(test_score), np.ones_like(ood_score)))
    total_scores = np.concatenate((test_score, ood_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, total_scores, pos_label=1)
    aps = average_precision_score(y_true, total_scores)

    best_threshold_arg = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_arg]

    y_class = (total_scores >= best_threshold)
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_class).ravel()
    tpr_new = tp / (tp + fn)
    fpr_new = fp / (fp + tn)
    tnr_new = tn / (tn + fp)
    gmean = np.sqrt(tpr_new * tnr_new)
    auroc_score = metrics.roc_auc_score(y_true, total_scores)

    return auroc_score, gmean, aps, tpr_new, fpr_new