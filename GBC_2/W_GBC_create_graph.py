import itertools
import math
import warnings
from GBC_2.wkmeans_no_random import WKMeans
from sklearn.cluster import k_means
from GBC_2.lwkmeans import lwkmeans
import numpy as np
warnings.filterwarnings('ignore')
def get_dm(hb, w):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = center - hb
    w_mat = np.tile(w, (num, 1))
    sq_diff_mat = diff_mat ** 2 * w_mat
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = sum(distances)
    if num > 1:
        return sum_radius / num
    else:
        return 1

def division(hb_list, hb_list_not, division_num, K, parent_w, num_neighbors):
    gb_list_new = []
    i = 0
    K = 2
    split_threshold = num_neighbors
    for hb in hb_list:
        hb_no_w = hb
        if len(hb_no_w) >= 2 * split_threshold:
            i = i + 1
            ball, child_w = spilt_ball_by_k(hb_no_w, parent_w, K, division_num)
            flag = 0
            for i in range(len(ball)):
                if len(ball[i]) == 0:
                    flag = 1
                    break
            if flag == 0:
                dm_child_ball = []
                child_ball_length = []
                dm_child_divide_len = []
                for i in range(K):
                    temp_dm = get_dm(np.delete(ball[i], -1,axis = 1), parent_w)
                    temp_len = len(ball[i])
                    dm_child_ball.append(temp_dm)
                    child_ball_length.append(temp_len)
                    dm_child_divide_len.append(temp_dm * temp_len)
                w0 = np.array(child_ball_length).sum()
                dm_child = np.array(dm_child_divide_len).sum() / w0
                dm_parent = get_dm(np.delete(hb_no_w,-1,axis = 1), parent_w)
                t2 = (dm_child < dm_parent)
                if t2:
                    for i in range(K):
                        gb_list_new.append(ball[i])
                else:
                    hb_list_not.append(hb)
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not

def KNN_ball_outer_2(ball_center_num, k, distances, ball_list_w):
    hb_graph = []
    hb_graph_w = []
    result_temp = []
    result = []
    for row in distances:
        sorted_indices = np.argsort(row)
        k_smallest_indices = sorted_indices[1:k]
        result_temp.append(k_smallest_indices)
    for re in result_temp:
        result.append([int(ball_center_num[i]) for i in re])
    for i in range(len(result)):
        perms = [(int(ball_center_num[i]), num) for num in result[i]]
        hb_graph.extend(perms)
    hb_graph = sorted(list(set(hb_graph)), key=lambda x: (x[0], x[1]))
    for i in range(len(hb_graph)):
        w = ball_list_w[0]
        hb_graph_w.extend([w])
    return hb_graph, hb_graph_w

def KNN_ball_outer(ball_center_num, k, ball_list_w, ball_center):
    hb_graph = []
    hb_graph_w = []
    if len(ball_center) <= k:
        id_num = [int(x) for x in np.array(ball_center_num)[:, -1]]
        perms = list(itertools.permutations(id_num, 2))
        hb_graph.extend(perms)
        for temp in perms:
            w = (ball_list_w[0] + ball_list_w[1]) / 2
            hb_graph_w.extend(w)
    else:
        distances = get_distance_w_2(ball_center_num, ball_list_w, ball_center)
        list_graph, list_graph_w = KNN_ball_outer_2(ball_center_num, k, distances, ball_list_w)
        hb_graph.extend(list_graph)
        hb_graph_w.extend(list_graph_w)
    return hb_graph, hb_graph_w

def KNN_ball_inner(hb_list_number, K, ball_list_w):
    hb_list = []
    hb_number =[]
    hb_graph =[]
    hb_graph_w =[]
    for hb_have_label in hb_list_number:
        temp_number = [int(temp[-1]) for temp in hb_have_label]
        hb_number.append(temp_number)
        hb_list.append(np.delete(hb_have_label, -1, axis=1))
    for i in range(len(hb_list_number)):
        if len(hb_list_number[i]) <= K:
            id_num = [int(x) for x in hb_list_number[i][:, -1]]
            perms = list(itertools.permutations(id_num, 2))
            hb_graph.extend(perms)
            for temp in perms:
                w = (ball_list_w[0] + ball_list_w[1]) / 2
                hb_graph_w.extend(w)
        else:
            distances = get_distance_w_2(hb_number[i], ball_list_w, hb_list[i])
            list_graph, list_graph_w = KNN_ball_outer_2(hb_number[i], K, distances, ball_list_w)
            hb_graph.extend(list_graph)
            hb_graph_w.extend(list_graph_w)
    return hb_graph, hb_graph_w

def load_graph_w_natural_v2(hb_list,ball_list_w, K, origin_data):
    graph_label = []
    hb_list_number = hb_list
    hb_graph_1, hb_graph_w1 = KNN_ball_inner(hb_list_number, K, ball_list_w)
    ball_center_num, ball_center, ball_center_no_idx = get_center_and_num(hb_list_number, ball_list_w)
    hb_graph_2, hb_graph_w2 = KNN_ball_outer(ball_center_num, K,  ball_list_w, ball_center_no_idx)
    hb_graph = hb_graph_1 + hb_graph_2
    hb_graph_w = hb_graph_w1 + hb_graph_w2
    graph_data = origin_data
    directed_graph_w_matrix = np.zeros((len(graph_data), len(graph_data)))
    for index, (i, j) in enumerate(hb_graph):
        w = hb_graph_w[index]
        dis = np.sum(np.linalg.norm(graph_data[i] - graph_data[j]) * w)
        directed_graph_w_matrix[i, j] = dis
    for i in range(len(directed_graph_w_matrix)):
        directed_graph_w_matrix[i, i] = np.sum(directed_graph_w_matrix[i, :]) / len(directed_graph_w_matrix[i, :])
    return directed_graph_w_matrix, graph_data, graph_label

def get_center_and_num(hb_list_number, ball_list_w):
    ball_center_num = []
    ball_center = []
    ball_center_no_idx = []
    for i, hb_ball in enumerate(hb_list_number):
        center = np.mean(hb_ball, axis=0)
        min = 100000000000
        for data in hb_ball:
            dis = np.sum(np.power((np.delete(data, -1, axis=0) - np.delete(center, -1, axis=0)), 2) * ball_list_w[i])
            if dis < min:
                data = np.array([data])
                idx = data[:, -1]
                center1 = data
                min = dis
        ball_center_num.append(idx)
        ball_center.append(center1)
        ball_center_no_idx.append(np.delete(center1, -1, axis=1))
    return ball_center_num, ball_center, ball_center_no_idx

def get_distance_w_2(ball_center_num, ball_list_w, ball_center):
    distances = np.zeros((len(ball_center_num), len(ball_center_num)))
    for i in range(len(ball_center_num)):
        for j in range(len(ball_center_num)):
            w = ball_list_w[0]
            distances[i, j] = np.sum(
                np.linalg.norm(ball_center[i] - ball_center[j]) * w)
    return distances

def spilt_ball_by_k(data, w, k, division_num):
    centroids = []
    data_no_label = np.delete(data, -1, axis=1)
    k = 2
    center = data_no_label.mean(0)
    p_max1 = np.argmax(((data_no_label - center) ** 2).sum(axis=1) ** 0.5)
    p_max2 = np.argmax(((data_no_label - data_no_label[p_max1]) ** 2).sum(axis=1) ** 0.5)
    c1 = (data_no_label[p_max1] + center) / 2
    c2 = (data_no_label[p_max2] + center) / 2
    centroids.append(c1)
    centroids.append(c2)
    idx = np.ones(len(data_no_label))
    for i in range(len(data_no_label)):
        subs = centroids - data_no_label[i, :]
        w_dimension2 = np.power(subs, 2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(k)
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    ball = []
    for i in range(k):
        ball.append(data[idx == i, :])
    return ball, w

def load_wkmeans_weight_and_ball(K, data):
    data_no_label = np.delete(data, -1, axis=1)
    centers = k_means(data_no_label, K, init="k-means++", n_init=10)[0]
    model = WKMeans(n_clusters=K, max_iter=10, belta=4, centers=centers)
    cluster = model.fit_predict(data_no_label)
    w = model.w
    ball = []
    for i in range(K):
        ball.append(data[cluster == i, :])
    return ball, w

def load_lwkmeans_weight_and_ball(K, data, m):
    if m <= 50:
        return load_wkmeans_weight_and_ball(K, data)
    data_no_label = np.delete(data, -1, axis=1)
    cluster, w, _, _ = lwkmeans(data_no_label, K, alpha=0.005)
    ball = []
    for i in range(K):
        ball.append(data[cluster == i, :])
    return ball, w

def hbc_v2(data_all, y, K, num_neighbors, origin_data):
    n, m = data_all.shape
    hb_list_not_temp = []
    division_num = 1
    ball, all_w = load_lwkmeans_weight_and_ball(K, data_all, m)
    hb_list_temp = ball
    while 1:
        ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
        division_num = division_num + 1
        hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp, division_num, K, all_w, num_neighbors)
        ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
        if ball_number_new == ball_number_old:
            hb_list_temp = hb_list_not_temp
            break
    ball_list = []
    for index, hb_ball in enumerate(hb_list_temp):
        if len(hb_ball) != 0:
            ball_list.append(hb_ball)
    hb_list_temp = ball_list
    dic_w = {}
    for index, hb_all in enumerate(hb_list_temp):
        dic_w[index] = all_w
    directed_graph_w_matrix, graph_data, graph_label = load_graph_w_natural_v2(hb_list_temp, dic_w, K, origin_data)
    graph_w_dis_numpy = (directed_graph_w_matrix + directed_graph_w_matrix.T) / 2
    return graph_w_dis_numpy, graph_data, graph_label, hb_list_temp, directed_graph_w_matrix

