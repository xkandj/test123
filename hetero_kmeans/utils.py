from functools import reduce
import operator
import numpy as np
from scipy.spatial.distance import pdist


def compute_centroids(data, labels_onehot):
    """
    根据标签计算聚类质心
    Args:
        data: 样本数n * 特征数m
        labels: 样本数n * 聚类数k, 经过one-hot化的标签 (例如第5个样本的标签为3, 那么labels[4, 3] == 1, labels[4]的其他元素为0.)

    Returns:
        np.array 聚类数k * 特征数m, k个聚类质心, 每个质心是簇内样本特征均值
    """
    num = np.sum(labels_onehot, axis=0).reshape(labels_onehot.shape[1], 1)
    return labels_onehot.T.dot(data) / num


def compute_sqdistance(data, centroids):
    """
    计算每个样本与每个质心的距离
    Args:
        data: 样本数 * 特征数
        centroids: 质心数 * 特征数
    Returns: 
        np.array 样本数 * 质心数, 每个元素代表样本与质心的距离平方
    """
    n, m = data.shape[0], data.shape[1]
    distance = []
    if len(centroids.shape) == 1:
        centroids = centroids.reshape(1, centroids.shape[0])
    for c in range(len(centroids)):
        sqdis = np.sum(np.square(centroids[c] - data), axis=1).reshape(n, 1)
        distance.append(sqdis)
    if len(distance) == 1:
        return distance[0]
    ret = np.concatenate(distance, axis=1)
    return ret


def compute_si_intra_inter_sqdist(data, labels, n_clusters):
    '''
    计算各个簇的簇内两两距离, 计算每个样本与簇外样本的距离
    Args:
        data: 样本数 * 特征数
        labels: 样本数 * 1
        n_clusters: 聚类数

    Returns:
        intra_sqdist: [[matrix(n, 1)]]各个簇的簇内两两距离
        inter_sqdist: [[{matrix(n, 1)}]]每个样本与簇外样本的距离
    '''
    intra_sqdist = []
    inter_sqdist = []
    for c in range(n_clusters):
        intra_sqdist_c = []  # list[flatten array(簇内不相似度), ]
        inter_sqdist_c = []  # list[dict{label: flatten array(簇间不相似度)}]
        data_c = data[labels == c]
        for i in range(len(data_c)):
            data_c_i, data_c_i_rest = data_c[i], np.delete(data_c, i, axis=0)
            if len(data_c_i_rest) == 0:
                intra_sqdist_c_i = np.zeros((1, 1))
            else:
                intra_sqdist_c_i = compute_sqdistance(data_c_i_rest, data_c_i).reshape(-1, 1)
            inter_sqdist_c_i = {}
            for m in range(n_clusters):
                if c == m:
                    continue
                inter_sqdist_i = compute_sqdistance(data[labels == m], data_c_i).reshape(-1, 1)
                inter_sqdist_c_i[m] = inter_sqdist_i
            intra_sqdist_c.append(intra_sqdist_c_i)
            inter_sqdist_c.append(inter_sqdist_c_i)
        intra_sqdist.append(intra_sqdist_c)
        inter_sqdist.append(inter_sqdist_c)
    return intra_sqdist, inter_sqdist


def compute_pairwise_dist(data):  # 内部
    '''

    Args:
        data: 数据，np.ndarray -> n * m

    Returns:
        sqdist: 两两点距离平方，flatten array -> [n * (n-1)] *1
    '''
    return pdist(data, metric='sqeuclidean')


def compute_intra_and_inter_dist(data, labels, centroids, n_clusters):
    '''

    Args:
        data: 数据，np.ndarray -> n * m
        labels: 标签，np.ndarray -> n * n_clusters
        centroids: 质心，np.ndarray -> n_clusters * m

    Returns:
        intra_sqdiameter: 簇内点到质心平方距离，list -> [flatten array] * n_clusters
        intra_sqdist: 簇内点两两距离平方，list -> [flatten array] * n_clusters
        inter_sqdist: 质心间两两距离平方，flatten array -> [n * (n-1)] * 1

    '''
    intra_sqdist = []
    intra_sqdiameter = []
    for c in range(n_clusters):
        data_c = data[labels == c]
        intra_sqdist_c = compute_pairwise_dist(data_c).reshape(-1, 1)
        intra_sqdist.append(intra_sqdist_c)
        intra_sqdiameter.append(np.sum(np.square(data_c - centroids[c]), axis=1).reshape(-1, 1))
    inter_sqdist = compute_pairwise_dist(centroids).reshape(-1, 1)
    return intra_sqdiameter, intra_sqdist, inter_sqdist


def compute_frobenius_norm(centroids, centroids_new):
    '''
    计算质心偏移
    Args:
        centroids: 质心数 * 特征数
        centroids_new: 质心数 * 特征数

    Returns:
        质心偏移总量: float
    '''
    center_shift = centroids - centroids_new
    return (center_shift ** 2).sum()


def clusters_onehot_transfer(labels, n_clusters):
    '''
    将标签labels按照n_clusters进行one-hot化
    Args:
        labels: 样本数 * 1
        n_clusters: 聚类数
    
    Returns:
        np array: 样本数 * n_clusters
    '''
    return np.eye(n_clusters)[labels]


def total_distance(*args):
    return reduce(operator.add, args)
