from abc import ABC
from io import StringIO
import pickle

import numpy as np
import pandas as pd

from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_algorithm import BaseAlgorithm

logger = get_fmpc_logger(__name__)


class KMeansAlgorithmBasePred(BaseAlgorithm, ABC):

    def do_ready(self):
        self.flnode_nid = self.job.flnodes[0].node.nid
        model = pickle.loads(self.file_system_client.read_bytes(self.model_input.url))
        self.numerical_features = model.get('numerical_features')
        self.centroids = model.get('best_centroids')
        self.dict_fillna = model.get('dict_fillna')
        self.task_type = self.job.extra.get("task_type")
        self.cat_features = [c.name for c in self.model_input.columns if c.is_cat_feature]

        if self.task_type == 'batch_predict':
            points_df = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)
            self.data_features = points_df[self.numerical_features].astype(float)
            pre_data_str = self.file_system_client.read_content(self.dataset_input.pre_dataset_remote_id)
            pre_data_df = pd.read_csv(StringIO(pre_data_str), float_precision='round_trip')
            self.data_features[self.dataset_input.match_column_list] = pre_data_df[self.dataset_input.match_column_list]

            self.df_fill_nan(self.data_features, self.numerical_features, self.cat_features)

            self.points = self.data_features[self.numerical_features].to_numpy()

        elif self.task_type == 'api_predict':
            # 从job_dict的meta中读取当前节点的原始api预测数据
            data_dict = self.model_input.all_nodes_params[self.curr_nid].param.meta.data

            # 转换成DataFrame
            df_data = pd.DataFrame.from_dict(data_dict).T #.reset_index().drop(columns=['index'])
            # 只拿对应于模型numerical_features的值 todo: 校验模型numerical_features与api预测的输入特征是否匹配 
            self.data_features = df_data[self.numerical_features].astype(float)

            self.df_fill_nan(self.data_features, self.numerical_features, self.cat_features)

            self.points = self.data_features.values
            self.uids = df_data.index.to_list()

    def df_fill_nan(self, df, all_features, cat_features):
        fdicttmp = {}
        for col in all_features:
            col_fillna = None
            if self.dict_fillna:
                col_fillna = self.dict_fillna.get(col)
            col_ = df[col]
            if col in cat_features:
                mode = col_.mode()
            else:
                mode = col_.round(2).mode()
            if len(mode) > 0:
                fdicttmp[col] = mode[0]
            else:
                fdicttmp[col] = 0
            if col_fillna is None:
                col_.fillna(fdicttmp[col], inplace=True)
            else:
                col_.fillna(col_fillna, inplace=True)

        if self.dict_fillna is None:
            self.dict_fillna = fdicttmp
