# encoding:utf-8
import pickle
from io import StringIO

import pandas as pd

from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_kmeans.k_means_predict_base import KMeansAlgorithmBasePred
from wares.hetero_kmeans.utils import compute_sqdistance

logger = get_fmpc_logger(__name__)


class KMeansHostPred(KMeansAlgorithmBasePred):
    """
        Attributes:
        flnode_nid(str): 合作节点的nid
        points(pd.DataFrame): 数值型特征numerical_features
        centroids(np.ndarray): 模型训练后生成的质心
    """

    def __init__(self, ware_id, **kwargs):
        super(KMeansAlgorithmBasePred, self).__init__(ware_id, **kwargs)
        self.flnode_nid = None
        self.points = None
        self.centroids = None

    def do_start(self):
        self.predict()

    def static_role(self) -> str:
        return Role.HOST

    def predict(self):
        sqdist_b = compute_sqdistance(self.points, self.centroids)
        self.algo_data_transfer.kmeans_single_predict_distance.send(sqdist_b, self.ctx, self.curr_nid)
