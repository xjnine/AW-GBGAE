from sklearn import metrics
from munkres import Munkres
import numpy as np
def cal_clustering_acc(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)
    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(true_label, new_predict)
    return acc

def cal_clustering_metric(truth, prediction):
    truth = np.array(truth).flatten()
    prediction = np.array(prediction).flatten()
    nmi = metrics.normalized_mutual_info_score(truth, prediction)
    acc = cal_clustering_acc(truth, prediction)
    ari = metrics.cluster.adjusted_rand_score(truth, prediction)
    return acc, nmi, ari
