import numpy as np
from GBC_2.W_GBC_create_graph import hbc_v2

def load_data_WGBC_v2(data, y, k, num_neighbors):
    origin_data = data
    n, m = data.shape
    y = np.arange(n).reshape((n, 1))
    data = np.hstack((data, y))
    graph_w_dis_numpy, graph_data, graph_label, hb_list_temp, directed_graph_w_matrix = hbc_v2(data, y, k, num_neighbors, origin_data)
    graph_label = y
    return graph_data, graph_label, graph_w_dis_numpy, hb_list_temp, directed_graph_w_matrix


