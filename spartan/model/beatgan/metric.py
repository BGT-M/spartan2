

import numpy as np
from sklearn import metrics


def evaluate(labels, scores, set_th=None):

    auc_roc = metrics.roc_auc_score(labels, scores)
    auc_prc = metrics.average_precision_score(labels, scores)
    precisions, recalls, ths = metrics.precision_recall_curve(labels, scores)
    f1 = 2/(1/precisions+1/recalls)
    best_f1 = np.max(f1)
    best_th = ths[np.argmax(f1)]

    # threshold f1
    if set_th is not None:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= set_th] = 1
        tmp_scores[tmp_scores < set_th] = 0
        print(metrics.classification_report(labels, tmp_scores))
        print(metrics.confusion_matrix(labels, tmp_scores))

    return auc_prc, auc_roc, best_th, best_f1
