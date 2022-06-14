import json
import pickle
import traceback
from io import StringIO

import numpy as np
import pandas as pd
from polars import DataFrame

from fmpc.fl.consts import Role
from fmpc.utils.report.ReportPrediction import PredictionService
from wares.hetero_kmeans.k_means_base import logger, PREDICT_DOWNLOAD
from wares.hetero_kmeans.k_means_predict_base import KMeansAlgorithmBasePred
from wares.hetero_kmeans.utils import compute_sqdistance


class KMeansGuestPred(KMeansAlgorithmBasePred):
    """
    Attributes:
        flnode_nid(str): 合作节点的nid
        uids(pd.Series): 数据集id
        points(pd.DataFrame): 数值型特征numerical_features
        centroids(np.ndarray): 模型训练后生成的质心
    """

    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)
        self.flnode_nid = None
        self.uids = None
        self.points = None
        self.centroids = None
        self.api_pred_results = {}

    def do_start(self):
        labels = self.predict()
        if self.task_type == 'batch_predict':
            df_pred = pd.concat((self.data_features[self.dataset_input.match_column_list], pd.DataFrame(labels, columns=['labels'])), axis=1)
            self.set_batch_predict_output(df_pred)
        elif self.task_type == 'api_predict':
            self.api_pred_results = self.prepare_api_result(labels)
            self.set_api_predict_output()

    def set_batch_predict_output(self, df_pred):

        predict_remote_id = self.file_system_client.write_content(df_pred.to_csv(index=False))
        logger.info("======= predict_remote_id :{} ".format(predict_remote_id))
        predict_res = {
            "name": "批量预测结果",
            "prefastdfs": predict_remote_id
        }
        self.ware_ctx.set_ware_output_data(PREDICT_DOWNLOAD, json.dumps(predict_res))

    def static_role(self) -> str:
        return Role.GUEST

    def predict(self):
        sqdist_a = compute_sqdistance(self.points, self.centroids)
        sqdist_b = self.algo_data_transfer.kmeans_single_predict_distance.get(self.listener, self.flnode_nid)
        return np.argmin(np.sqrt(sqdist_a + sqdist_b), axis=1)

    def prepare_api_result(self, labels):
        api_pred_results = {}
        for uid, label in zip(self.uids, labels):
            api_pred_results[uid] = {}
            api_pred_results[uid]['fed_code'] = 0
            api_pred_results[uid]['fed_message'] = "success"
            api_pred_results[uid]['data'] = {}
            api_pred_results[uid]['data']['ypred'] = int(label)
            api_pred_results[uid]['data']['ypred_prob'] = -1
        return api_pred_results

    def ex_callback(self, ex):
        api_pred_results = {}
        # 用第一个uid表示该批次预测，存在预测错误的情况
        uid = self.uids[0]
        api_pred_results[uid] = {}
        api_pred_results[uid]['fed_code'] = 50001
        api_pred_results[uid]['fed_message'] = str(ex)
        api_pred_results[uid]['data'] = {}
        api_pred_results[uid]['data']['ypred'] = -1
        api_pred_results[uid]['data']['ypred_prob'] = -1
        return api_pred_results

    def set_api_predict_output(self):
        try:
            logger.info("========== 正常回调 ==========")
            PredictionService.report_result(self.job_id, self.api_pred_results)
        except Exception as ex:
            logger.error('模型API预测出错了，error={}'.format(str(ex)))
            self.log_error('模型API预测出错了，error={}'.format(str(ex)))
            logger.info("========== 异常回调 ==========")
            PredictionService.report_result(self.job_id, self.ex_callback(ex))
            raise ex