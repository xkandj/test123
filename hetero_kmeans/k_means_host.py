import numpy as np

from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger

from wares.hetero_kmeans.k_means_base import KMeansAlgorithmBase
from wares.hetero_kmeans.utils import clusters_onehot_transfer, compute_centroids, compute_frobenius_norm,\
    compute_intra_and_inter_dist, compute_si_intra_inter_sqdist, compute_sqdistance

logger = get_fmpc_logger(__name__)


class KMeansHost(KMeansAlgorithmBase):

    def kmeans_single(self, data, n_init, seed):
        centroids = self.init_centroids(data, n_init, seed)
        self.log_info(f"[n_init -> {n_init}]kmeans++ init centroids...")

        i = 0
        center_shift = None
        inertia = None
        relocate_labels = None
        while not self.early_stop(i, self.max_iter, center_shift, self.tol):
            distance_b = compute_sqdistance(data, centroids)
            self.algo_data_transfer.kmeans_single_distance.send(
                distance_b, self.ctx, self.curr_nid, n_init, i)

            relocate_labels = self.algo_data_transfer.kmeans_single_relocate_labels.get(
                self.listener, self.guest_nid, n_init, i)
            labels = clusters_onehot_transfer(relocate_labels, self.n_clusters)
            centroids_new = compute_centroids(data, labels)
            center_shift_b = compute_frobenius_norm(centroids, centroids_new)
            self.algo_data_transfer.kmeans_single_center_shift_b.send(
                center_shift_b, self.ctx, self.curr_nid, n_init, i)
            centroids = centroids_new
            center_shift, inertia = self.algo_data_transfer.kmeans_single_center_shift_all.get(
                self.listener, self.guest_nid, n_init, i)
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

        c_b = data[idx0]
        c_idx_list = [idx0, ]
        for k_cluster in range(self.n_clusters - 1):
            dist_c_b = compute_sqdistance(data, c_b)
            self.algo_data_transfer.kmeans_plusplus_init_distance.send(
                dist_c_b, self.ctx, self.curr_nid, n_init, k_cluster)
            c_idx = self.algo_data_transfer.kmeans_plusplus_init_idx.get(
                self.listener, self.guest_nid, n_init, k_cluster)
            c_idx_list.append(c_idx)

        return data[c_idx_list]

    def static_role(self) -> str:
        return Role.HOST

    def evaluate(self, data_features, best_inertia, best_centroids, best_labels):
        # intra_sqdiameter, intra_sqdist, inter_sqdist
        intra_sqdiameter_b_list, intra_sqdist_b_list, inter_sqdist_b_list = compute_intra_and_inter_dist(
                                                                    data_features, best_labels, best_centroids, self.n_clusters)
        intra_sqdist_bi, inter_sqdist_bi = compute_si_intra_inter_sqdist(data_features, best_labels, self.n_clusters)
        self.algo_data_transfer.kmeans_evaluate_intra_and_inter_dist_b.send(
            (intra_sqdiameter_b_list, intra_sqdist_b_list, inter_sqdist_b_list, intra_sqdist_bi, inter_sqdist_bi), self.ctx, self.curr_nid)
        body = None
        if self.is_owner:
            body = self.algo_data_transfer.kmeans_model_report_body.get(self.listener)
        return body

    def predict_for_model_eval(self):
        sqdist_b = compute_sqdistance(self.data_features.values, self.centroids)
        self.algo_data_transfer.kmeans_single_distance.send(sqdist_b, self.ctx, self.curr_nid, 'model_eval', '')
        labels = self.algo_data_transfer.kmeans_single_relocate_labels.get(self.listener, self.guest_nid, 'model_eval', '')
        return labels

