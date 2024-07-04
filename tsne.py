import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

def plot_tsne(data, label, key, K, str_label):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    col_max = result.max(axis=0)
    col_min = result.min(axis=0)
    str = str_label + " of " + key + " dataset"
    plot_embedding(result, label, str, K)
    xx = (col_max[0] - col_min[0]) / 5
    yy = (col_max[1] - col_min[1]) / 5
    plt.xlim(col_min[0] - xx, col_max[0] + xx)
    plt.ylim(col_min[1] - yy, col_max[1] + yy)
    plt.legend().remove()
    filename = 'D:\cluster\plus\W-GBC-fin\img' + '\\' + key
    if not os.path.exists(filename):
        os.makedirs(filename)
    print(str)
    plt.show()
    plt.savefig(filename + '\\' + str + '.png')

color = {
        0:'#DC143C',#枣红色
        1:'#4169E1',
        2: '#16ccd0',#浅蓝色
        3: '#ed7231',
        4:'#87CEFA',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#ff8444',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444', }
marker = {
        0:'s',
        1: '^',
        2: 'o',
        3: 'v',
    }

def plot_embedding(data, label, title,K):
    fig = plt.figure()
    data_list = []
    for i in range(K):
        index = np.where((label == i) | (label == i+K))
        data_list.extend(data[index,:])
    for i in range(K):
        plt.scatter(data_list[i][:, 0], data_list[i][:, 1], s=7, c=color.get(i), linewidths=4, alpha=1, marker=marker.get(i%K), label=i)
    plt.rcParams['legend.fontsize'] = 13
    plt.legend()
    return fig

def plot_tsne_3(data, label, key, K, str_label):
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-150, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-150, 100])
    data_list = []
    for i in range(K):
        index = np.where((label == i) | (label == i + K))
        data_list.extend(X_tsne[index, :])
    for i in range(K):
        ax.scatter(data_list[i][:, 0], data_list[i][:, 1], data_list[i][:, 2], s=7, c=color.get(i), linewidths=4, alpha=1,
                    marker=marker.get(i % K), label=i)

    plt.rcParams['legend.fontsize'] = 13
    plt.legend()
    filename = 'D:\cluster\plus\W-GBC-fin\img-3' + '\\' + key
    if not os.path.exists(filename):
        os.makedirs(filename)
    str = str_label + " of " + key + " dataset"
    print(str)
    plt.show()

def plot_umap(data, label, key, K, str_label):
    print('Computing t-SNE embedding')
    reducer = umap.UMAP(random_state=42)
    reducer.fit(data)
    embedding = reducer.transform(data)
    col_max = embedding.max(axis=0)
    col_min = embedding.min(axis=0)
    str = str_label + " of " + key + " dataset"
    plot_embedding(embedding, label, str, K)
    xx = (col_max[0] - col_min[0]) / 10
    yy = (col_max[1] - col_min[1]) / 10
    plt.xlim(col_min[0] - xx, col_max[0] + xx)
    plt.ylim(col_min[1] - yy, col_max[1] + yy)
    plt.show()