import collections
import math
import operator
from functools import reduce

import numpy as np
from scipy.spatial.distance import squareform

from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger

from wares.hetero_kmeans.k_means_base import KMeansAlgorithmBase
from wares.hetero_kmeans.utils import clusters_onehot_transfer, compute_centroids, compute_frobenius_norm, \
    compute_intra_and_inter_dist, compute_si_intra_inter_sqdist, compute_sqdistance, \
    total_distance

logger = get_fmpc_logger(__name__)


class KMeansGuest(KMeansAlgorithmBase):

    def kmeans_single(self, data, n_init, seed):
        centroids = self.init_centroids(data, n_init, seed)
        self.log_info(f"[n_init -> {n_init}]kmeans++ init centroids...")

        i = 0
        center_shift = None
        inertia = None
        relocate_labels = None
        while not self.early_stop(i, self.max_iter, center_shift, self.tol):
            distance_a = compute_sqdistance(data, centroids)
            distance_b = 0
            for flnode_nid in self.flnode_nid_list:
                distance_b += self.algo_data_transfer.kmeans_single_distance.get(
                    self.listener, flnode_nid, n_init, i)
            total = total_distance(distance_a, distance_b)  # 总距离
            relocate_labels = gen_labels_and_relocate_empty_cluster(total, self.n_clusters)
            self.algo_data_transfer.kmeans_single_relocate_labels.send(
                relocate_labels, self.ctx, self.curr_nid, n_init, i)
            labels_onehot = clusters_onehot_transfer(relocate_labels, self.n_clusters)
            inertia = (total * labels_onehot).sum()
            centroids_new = compute_centroids(data, labels_onehot)

            center_shift_a = compute_frobenius_norm(centroids, centroids_new)
            for flnode_nid in self.flnode_nid_list:
                center_shift_b = self.algo_data_transfer.kmeans_single_center_shift_b.get(
                    self.listener, flnode_nid, n_init, i)
            centroids = centroids_new
            center_shift = center_shift_a + center_shift_b
            self.algo_data_transfer.kmeans_single_center_shift_all.send(
                (center_shift, inertia), self.ctx, self.curr_nid, n_init, i)

            self.log_info(f'[n_init -> {n_init}][iter -> {i}] inertia -> {inertia}, center_shift -> {center_shift}')
            i += 1
        return inertia, centroids, relocate_labels

    def kmeans_plusplus(self, data, n_init, seed):
        """

        Args:
            data:
            n_clusters:
            n_init:
            seed:

        Returns:

        """
        np.random.seed(seed)
        idx0 = np.random.randint(0, data.shape[0])

        c_a = data[idx0]
        acc_dist = 0
        c_idx_list = [idx0, ]
        for k_cluster in range(self.n_clusters - 1):
            dist_c_a = compute_sqdistance(data, c_a)
            dist_c_b = 0
            for flnode_nid in self.flnode_nid_list:
                dist_c_b += self.algo_data_transfer.kmeans_plusplus_init_distance.get(
                    self.listener, flnode_nid, n_init, k_cluster)
            total_dist = dist_c_a + dist_c_b
            acc_dist += total_dist
            c_idx = np.argmax(acc_dist)
            acc_dist[c_idx] = 0
            c_idx_list.append(c_idx)
            c_a = data[c_idx]
            self.algo_data_transfer.kmeans_plusplus_init_idx.send(c_idx, self.ctx, self.curr_nid, n_init, k_cluster)

        return data[c_idx_list]

    def static_role(self) -> str:
        return Role.GUEST

    def evaluate(self, data_features, best_inertia, best_centroids, best_labels):
        # 前置计算一些用到的距离
        n_samples = len(best_labels)
        intra_sqdiameter_a, intra_sqdist_a, inter_sqdist_a = compute_intra_and_inter_dist(
            data_features, best_labels, best_centroids, self.n_clusters)
        intra_sqdist_ai, inter_sqdist_ai = compute_si_intra_inter_sqdist(data_features, best_labels, self.n_clusters)
        # intra_sqdiameter_b_list, intra_sqdist_b_list, inter_sqdist_b_list, intra_inter_sqdist_b_list
        intra_sqdiameter_b_list, intra_sqdist_b_list, inter_sqdist_b_list = [], [], []
        intra_sqdist_bi_list, inter_sqdist_bi_list = [], []
        for flnode_nid in self.flnode_nid_list:
            intra_sqdiameter, intra_sqdist, inter_sqdist, intra_sqdist_i, inter_sqdist_i = \
                self.algo_data_transfer.kmeans_evaluate_intra_and_inter_dist_b.get(self.listener, flnode_nid)
            intra_sqdiameter_b_list.append(intra_sqdiameter)
            intra_sqdist_b_list.append(intra_sqdist)
            inter_sqdist_b_list.append(inter_sqdist)
            intra_sqdist_bi_list.append(intra_sqdist_i)
            inter_sqdist_bi_list.append(inter_sqdist_i)
        # intra_diameter, intra_dist
        intra_diameter, intra_dist = _compute_intra_distance_and_diameter(
            intra_sqdist_a, intra_sqdist_b_list, intra_sqdiameter_a, intra_sqdiameter_b_list, self.n_clusters)
        # inter_dist
        inter_dist = np.sqrt(inter_sqdist_a + np.sum(np.concatenate(inter_sqdist_b_list, axis=1),axis=1).reshape(-1,1))
        # Dunn Index
        dunn_index = compute_dunn_index(intra_dist, inter_dist, self.n_clusters)
        # DB Index
        db_index = compute_davies_bouldin_score(intra_diameter, inter_dist, self.n_clusters)
        # weighted_avg_radius
        cluster_radius_array = np.zeros((self.n_clusters,))
        for c in range(self.n_clusters):
            if intra_dist[c].size == 0:
                cluster_radius_array[c] = 0
            else:
                cluster_radius = math.sqrt(np.max(intra_dist[c])) / 2
                cluster_radius_array[c] = cluster_radius
        weighted_avg_radius = 0.0
        for c in range(self.n_clusters):
            weighted_avg_radius += (best_labels==c).sum() / n_samples * cluster_radius_array[c]
        # Silhouette Coefficient
        si, min_inter_cluster_distances = compute_silhouette_coefficient(
            intra_sqdist_ai, inter_sqdist_ai, intra_sqdist_bi_list, inter_sqdist_bi_list , n_samples, self.n_clusters)

        # report dict
        key_index = {'DBScore': db_index,
                     'silhouetteScore': si,
                     'DunnIndex': dunn_index,
                     'nClusters': self.n_clusters,
                     'inertia': weighted_avg_radius}  # todo  inertia字段命名不准确，这里是簇半径的加权平均

        centroid_info = {
            'centroidDistances': squareform(inter_dist.squeeze()).tolist(),
            'minInterClusterDistances': min_inter_cluster_distances
        }

        train_detail = _labels_detail(best_labels)

        labels_detail_list = [{'label': str(i),
                               'trainNumber': train_detail[0][i],
                               'trainPercent': train_detail[1][i],
                               } for i in train_detail[0]]
        labels_detail_list = sorted(labels_detail_list, key=lambda x: int(x['label']))

        if self.task_type == 'model_train':
            mean_and_variance = cal_mean_and_variance(self.data_features_sampled, best_labels, self.n_clusters)
        else:
            mean_and_variance = cal_mean_and_variance(self.data_features, best_labels, self.n_clusters)

        assessment = {'keyIndex': key_index,
                'centroidInfo': centroid_info,
                'labelsDetails': labels_detail_list,
                'meanAndVariance': mean_and_variance
                }

        eval_report = {
            "modelReportType": "unsupervised:cluster", 
            "name": "模型报告", 
            "assessment": assessment
            }

        if not self.is_owner:
            self.algo_data_transfer.kmeans_model_report_body.send(eval_report, self.ctx)

        return eval_report

    def predict_for_model_eval(self):
        sqdist_a = compute_sqdistance(self.data_features.values, self.centroids)
        sqdist_b = 0
        for flnode_nid in self.flnode_nid_list:
            sqdist_b += self.algo_data_transfer.kmeans_single_distance.get(self.listener, flnode_nid, 'model_eval', '')
        total = total_distance(sqdist_a, sqdist_b)  # 总距离
        labels = np.argmin(np.sqrt(total), axis=1)
        self.algo_data_transfer.kmeans_single_relocate_labels.send(labels, self.ctx, self.curr_nid, 'model_eval', '')
        return labels


def gen_labels_and_relocate_empty_cluster(total, n_clusters):
    '''
    生成标签:
        根据最小距离确定labels, 统计各个类的样本数量
    填充空类:
        空类数大于1时, 
            报错
        空类数等于1时, 
            计算所有样本与各个质心的距离总和
            按照距离总和排序, 从距离最大的样本开始, 
                如果此样本所在类的样本数不是1,
                    将此样本划归给空类, 返回labels
                否则,
                    循环到下一个样本
    
    Args:
        total: np array 样本数 * 聚类数
        n_clusters: 聚类数

    Return:
        labels: np array 样本数 * 1
    '''
    # 最小距离对应的列号为样本标签
    labels = np.argmin(total, axis=1)  # (n, )
    # 分类统计标签数量
    labels_count = np.bincount(labels)  # (max_value+1, )
    # 与n_clusters个数对齐
    differ_k = n_clusters - len(labels_count)
    if differ_k != 0:
        labels_count = np.hstack((labels_count, np.zeros(differ_k)))
    # 标签数量为零的空集总数
    empty_count = sum(labels_count == 0)  
    if empty_count > 1:
        raise ValueError(f'空类数量大于1，K值可能过大。empty_count -> {empty_count}')
    elif empty_count == 1:
        empty_indice = np.argwhere(labels_count == 0)  # 空集的类
        total_dist = np.sum(total, axis=1)  # n个数据点到k质点的距离总和
        sorted_dist = np.argsort(total_dist)  # 升序排列的总距离索引
        for i in sorted_dist[::-1]:
            if labels_count[labels[i]] > 1:  # 最远距离的点所在的类的集合数量大于1
                labels[i] = empty_indice  # 替换该点的类
                return labels
    else:
        return labels


def _compute_intra_distance_and_diameter(intra_sqdist_a, intra_sqdist_b_list, intra_sqdiameter_a, intra_sqdiameter_b_list, n_clusters):
    len_b = len(intra_sqdist_b_list)
    intra_dist = []
    intra_diameter = []
    for c in range(n_clusters):
        intra_sqdist_b_c_list = []
        intra_sqdiameter_b_c_list = []
        for b in range(len_b):
            intra_sqdist_b_c_list.append(intra_sqdist_b_list[b][c])
            intra_sqdiameter_b_c_list.append(intra_sqdiameter_b_list[b][c])
        intra_sqdist_b_c = np.sum(np.concatenate(intra_sqdist_b_c_list, axis=1), axis=1).reshape(-1,1)
        intra_dist.append(np.sqrt(intra_sqdist_a[c] + intra_sqdist_b_c))
        intra_sqdiameter_b_c = np.sum(np.concatenate(intra_sqdiameter_b_c_list, axis=1), axis=1).reshape(-1,1)
        one = intra_sqdiameter_a[c]
        intra_diameter.append(np.sqrt(one + intra_sqdiameter_b_c).sum() / len(one))
    return intra_diameter, intra_dist


def compute_si(intra_sqdist_ai, inter_sqdist_ai, intra_sqdist_bi_list, inter_sqdist_bi_list, n, n_clusters):
    si = 0
    closest_inter = collections.defaultdict(dict)
    len_b = len(intra_sqdist_bi_list)
    for c in range(n_clusters):
        intra_sqdist_ai_c = intra_sqdist_ai[c]
        inter_sqdist_ai_c = inter_sqdist_ai[c]
        c_i_num = len(intra_sqdist_ai_c)  # 第c类点的数量
        if c_i_num == 0:
            for cc in range(n_clusters):
                closest_inter[c][cc] = str(np.nan)
            continue
        c_i_rest_num = n - c_i_num
        for i in range(c_i_num):
            intra_sqdist_ai_c_i = intra_sqdist_ai_c[i]
            intra_sqdist_bi_c_i_list = []
            inter_sqdist_bi_c_i_list = []
            for b in range(len_b):
                intra_sqdist_bi_c_i_list.append(intra_sqdist_bi_list[b][c][i])
                inter_sqdist_bi_c_i_list.append(inter_sqdist_bi_list[b][c][i])
            intra_sqdist_bi_c_i = np.sum(np.concatenate(intra_sqdist_bi_c_i_list, axis=1), axis=1).reshape(-1,1)
            intra_dist_c_i = np.sqrt(intra_sqdist_ai_c_i + intra_sqdist_bi_c_i) # ai array
            ai = _calc_si_ai(c_i_num, intra_dist_c_i)
            inter_sqdist_ai_c_i = inter_sqdist_ai_c[i]
            bi = _calc_si_bi_and_update_closest_inter(
                c, inter_sqdist_ai_c_i, inter_sqdist_bi_c_i_list, closest_inter, c_i_rest_num, n_clusters)
            if c_i_num == 1: # 若某类只有一个点，si -> 0
                continue
            else:
                si += (bi - ai) / max(bi, ai)
    si /= n
    return si, closest_inter


def compute_silhouette_coefficient(intra_sqdist_ai, inter_sqdist_ai, intra_sqdist_bi_list, inter_sqdist_bi_list, n_samples, n_clusters):
    '''

    Args:
        intra_sqdist_ai:
        inter_sqdist_ai:
        intra_sqdist_bi_list:
        inter_sqdist_bi_list:
        n_samples:

    Returns:
        si: 轮廓系数
        min_inter_cluster_distances: 簇间最近距离
    '''
    if n_clusters > 1:
        si, closet_inter_dist_c_ij = compute_si(intra_sqdist_ai, inter_sqdist_ai, \
                                                        intra_sqdist_bi_list, inter_sqdist_bi_list, n_samples, n_clusters)
        min_inter_cluster_distances = []
        for i_, inner_dict in closet_inter_dist_c_ij.items():
            min_inter_cluster_distances.append([])
            for j_, val_ in inner_dict.items():
                min_inter_cluster_distances[i_].append(val_)
    else:
        si = None
        min_inter_cluster_distances = [[0]]
    return si, min_inter_cluster_distances


def _calc_si_ai(c_i_num, intra_dist_c_i):
    if c_i_num == 1:
        ai = 0
    else:
        ai = np.sum(intra_dist_c_i) / (c_i_num - 1)
    return ai


def _calc_si_bi_and_update_closest_inter(c, inter_sqdist_ai_c_i, inter_sqdist_bi_c_i_list, closest_inter, c_i_rest_num, n_clusters):
    bi = 0
    len_b = len(inter_sqdist_bi_c_i_list)
    for c_ in range(n_clusters):
        if c_ != c:
            if inter_sqdist_ai_c_i[c_].shape[0] == 0:
                closest_inter[c][c_] = str(np.nan)
                continue
            inter_sqdist_bi_c_i = inter_sqdist_bi_c_i_list[0]
            for b in range(1, len_b):
                inter_sqdist_bi_c_i[c_] += inter_sqdist_bi_c_i_list[b][c_]
            inter_dist_i = np.sqrt(inter_sqdist_ai_c_i[c_] + inter_sqdist_bi_c_i[c_])
            inter_dist_c_i = np.sum(inter_dist_i)
            bi += inter_dist_c_i
            dist = np.min(inter_dist_i)
            if c not in closest_inter.keys() or c_ not in closest_inter[c] or dist < closest_inter[c][c_]:
                closest_inter[c][c_] = dist
        else:
            closest_inter[c][c_] = 0
    bi /= c_i_rest_num
    return bi


def compute_dunn_index(intra_dist, inter_dist, n_clusters):
    intra_dist_all = np.vstack(intra_dist)
    if n_clusters > 1:
        dunn_index = np.min(inter_dist) / np.max(intra_dist_all)
    else:
        dunn_index = None
    return dunn_index


def compute_davies_bouldin_score(intra_diameter, inter_dist, n_clusters):
    db_index = 0
    inter_dist = squareform(inter_dist.squeeze())
    for i in range(n_clusters):
        if np.isnan(intra_diameter[i]): continue
        max_res = -np.inf
        for j in range(n_clusters):
            if i == j: continue
            if np.isnan(intra_diameter[j]): continue
            res = (intra_diameter[i] + intra_diameter[j]) / inter_dist[i][j]
            if res > max_res:
                max_res = res
        db_index += max_res
    db_index /= n_clusters
    return db_index


def cal_mean_and_variance(df, labels, n_clusters):
    mean_and_variance = []
    for col_name in df:
        mean_list = []
        variance_list = []
        for i in range(n_clusters):
            mean_value = df[col_name][labels==i].mean()
            variance_value = df[col_name][labels==i].std()
            if np.isnan(mean_value):
                mean_value = str(mean_value)
            if np.isnan(variance_value):
                variance_value = str(variance_value)   

            mean_list.append(mean_value)
            variance_list.append(variance_value)
        dd = {
            'name':col_name,
            'mean':mean_list,
            'variance':variance_list
        }
        mean_and_variance.append(dd)
    return mean_and_variance


def _labels_detail(labels):
    counter = collections.Counter(labels)
    tot = sum(counter.values())
    percent = {}
    for i in counter:
        percent[i] = counter[i] / tot
    return counter, percent
