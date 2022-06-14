import abc
import json
import pickle
import random
from abc import ABC
from io import StringIO

import numpy as np
import pandas as pd
from fmpc.fl.consts.fmpc_enums import FmpcDatasetColumnType

from fmpc.utils.EnvUtil import EnvUtil
from fmpc.utils.FastDfsSliceUtils import get_content, save_bytes, save_content
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_algorithm import BaseAlgorithm
from wares.common.ware_param import ModelNodeParam, ModelMeta, ModelWareParam, DatasetMetaColumn

logger = get_fmpc_logger(__name__)

CLUSTER_MODEL_REPORT = "CLUSTER_MODEL_REPORT"
DATASET_INPUT = 'input1'
MODEL_OUTPUT = 'HETERO_MODEL'
DATASET_OUTPUT = 'LABELED_DATASET'
EVALUATE_SAMPLES_COUNT = 10000
PREDICT_DOWNLOAD = "PREDICT_DOWNLOAD"
EVAL_PRED_DOWNLOAD = "REPORT_CSV_PRED"


class KMeansAlgorithmBase(BaseAlgorithm, ABC):

    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)
        # settings
        self.n_init = None
        self.n_clusters = None
        self.max_iter = None
        self.tol = None
        # dataset: dataframe
        #  self.parallel
        self.parallel = False  # todo  现在并行不通，算法ctx会拿走并不属于自己的事件
        available_cpu_count = EnvUtil.get_available_cpu_count()
        if available_cpu_count >= 4:
            cpu_pool_size = available_cpu_count // 2
        else:
            cpu_pool_size = available_cpu_count
        self.n_jobs = cpu_pool_size
        self.guest_nid = self.role_parser.get_nodes('GUEST')[0].node.nid
        self.dict_fillna = None

    def parse_settings(self):
        settings = self.job.settings
        self.n_init = settings.get('nInit')
        self.n_clusters = settings.get('nClusters')
        self.max_iter = settings.get('maxIter')
        self.tol = settings.get('tol')
        # init用的种子 todo 种子设置
        self.seed = 42

    def load_csv(self):
        df = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)
        return df

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

    def do_ready(self) -> None:
        super().do_ready()
        # 判断参与方数量，大于2报错
        if len(self.job.flnodes) > 1:
            raise ValueError("k-means不支持三方或无数据方学习")
        # common_params
        self.flnode_nid_list = [flnode.node.nid for flnode in self.job.flnodes]

        if self.job.extra is not None:
            self.task_type = self.job.extra.get('task_type')
        else:
            self.task_type = None

        if self.task_type != 'model_evaluate':
            self.parse_settings()
            # cat, con features
            self.cat_features = [c.name for c in self.dataset_input.columns if c.is_cat_feature]
            self.con_features = [c.name for c in self.dataset_input.columns if c.is_con_feature]
            numeric_types = FmpcDatasetColumnType.get_numeric_types()
            numerical_features = []
            for c in self.dataset_input.columns:
                if (c.type in numeric_types) and (not c.is_match_column) and (not c.is_target_column):
                    numerical_features.append(c.name)
            self.numerical_features = numerical_features
            if not self.numerical_features:
                raise IndexError(f"没有数值型特征，无法进行训练")
        else:
            self.cat_features = [c.name for c in self.model_input.columns if c.is_cat_feature]
            self.centroids, self.numerical_features, self.n_clusters, self.dict_fillna = self.load_model()
        
        dataset = self.load_csv()
        self.data_all = dataset
        self.data_features = dataset[self.numerical_features].astype(float)
        self.df_fill_nan(self.data_features, self.numerical_features, self.cat_features)

    def do_start(self):
        if self.task_type != 'model_evaluate':
            self.model_train()
        else:
            self.model_evaluate()

    def model_train(self):
        best_inertia = None
        best_labels = None
        best_centroids = None
        rand = random.Random(self.seed)
        n_init_seed = [rand.randint(0, 2 ** 32) for _ in range(self.n_init)]
        self.log_info(f"kmeans start... role -> {self.role_parser.get_current_node_role()}")
        X = self.data_features.values
        for n in range(self.n_init):
            inertia, centroids, labels = self.kmeans_single(X, n, n_init_seed[n])
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
        n_samples = len(best_labels)
        if n_samples > 2 * EVALUATE_SAMPLES_COUNT:
            idx_arr = equal_frequency_sample(best_labels, EVALUATE_SAMPLES_COUNT, self.seed)
            sampled_x = X[idx_arr]
            sampled_label = best_labels[idx_arr]
            self.data_features_sampled = self.data_features.iloc[idx_arr,:]
        else:
            sampled_x = X
            sampled_label = best_labels
            self.data_features_sampled = self.data_features
        # 评估并生成报告
        self.log_info(f"finish n_init kmeans... best_inertia -> {best_inertia}")
        logger.info(f"finish n_init kmeans... \nbest_inertia -> {best_inertia}\nbest_centroids -> {best_centroids}")
        report = self.evaluate(sampled_x, best_inertia, best_centroids, sampled_label)
        if self.is_owner:
            json_report = json.dumps(report)
            self.ware_ctx.set_ware_output_data(CLUSTER_MODEL_REPORT, json_report)

        self.save_model(best_centroids, self.numerical_features, self.n_clusters, self.dict_fillna)
        self.save_labeled_dataset(self.data_all, best_labels, DATASET_OUTPUT)

    def model_evaluate(self):

        pre_data_str = self.file_system_client.read_content(self.dataset_input.pre_dataset_remote_id)
        pre_data_df = pd.read_csv(StringIO(pre_data_str), float_precision='round_trip')
        pre_uids = pre_data_df[self.dataset_input.ori_column_id]
        column_id = self.data_all[self.dataset_input.column_id]

        predicted_labels = self.predict_for_model_eval()
        model_eval_report = self.evaluate(self.data_features.values, None, self.centroids, predicted_labels)
        if self.is_owner:
            model_eval_report_json = json.dumps(model_eval_report)
            self.ware_ctx.set_ware_output_data(CLUSTER_MODEL_REPORT, model_eval_report_json)
            
            df_predicted_labels = pd.DataFrame(predicted_labels, columns=['labels'])
            pred_test = pd.concat((pre_uids, column_id, df_predicted_labels), axis=1)
            predict_remote_id = self.file_system_client.write_content(pred_test.to_csv(index=False))
            eval_pred_res = {
                    "name": "评估预测结果",
                    "prefastdfs": predict_remote_id
                }
            eval_pred_res_json = json.dumps(eval_pred_res)
            self.ware_ctx.set_ware_output_data(EVAL_PRED_DOWNLOAD, eval_pred_res_json)

    def predict_for_model_eval(self):
        raise NotImplementedError

    def evaluate(self, data_features, best_inertia, best_centroids, best_labels):
        raise NotImplementedError

    def kmeans_single(self, data, n_init, seed):
        raise NotImplementedError

    def init_centroids(self, data, n_init, seed):
        return self.kmeans_plusplus(data, n_init, seed)

    def kmeans_plusplus(self, data, n_init, seed):
        raise NotImplementedError

    def early_stop(self, i, max_iter, center_shift, tol):
        if center_shift is None:
            return False
        elif center_shift <= tol:
            return True
        else:
            return i >= max_iter

    def save_model(self, best_centroids, numerical_features, n_clusters, dict_fillna):
        model = {
            'best_centroids': best_centroids,
            'numerical_features': numerical_features,
            'n_clusters': n_clusters,
            'dict_fillna': dict_fillna
        }

        return self.set_model_ware_output(model)

    def load_model(self):
        model = pickle.loads(self.file_system_client.read_bytes(self.model_input.url))
        centroids = model.get('best_centroids')
        numerical_features = model.get('numerical_features')
        n_clusters = model.get('n_clusters')
        dict_fillna = model.get('dict_fillna')

        return centroids, numerical_features, n_clusters, dict_fillna

    def set_model_ware_output(self, model_result):
        logger.info("--------------------< START KMeans Model Save >--------------------")
        self.log_info('--* START KMeans Model Save *--')
        model_remote_id = save_bytes(pickle.dumps(model_result))
        # 生成模型结果上报平台
        dataset_in = self.runtime_inputs[DATASET_INPUT]
        if dataset_in is None:
            model_output = ModelNodeParam.create_empty(self.job.currnode.node, 'NO_DATA_OWNER')
            return self.set_output(MODEL_OUTPUT, model_output)
        model_output = ModelNodeParam.create_empty(self.job.currnode.node, dataset_in.node_type)
        if dataset_in.param is not None:
            dataset_in_meta = dataset_in.param.meta
            model_meta = ModelMeta(dataset_in_meta.columns, dataset_in_meta.dataset_tag, dataset_in_meta.match_column,
                                   dataset_in.rules, dataset_in_meta.dataset_type, None, self.job.ware_version)
            model_output.param = ModelWareParam(model_remote_id, self.ware_id, model_meta, self.job_id,
                                                self.ware.cn_name, "", self.job.inputs[0].param_name_cn)
        self.set_output(MODEL_OUTPUT, model_output)

    def save_labeled_dataset(self, data_all, best_labels, download_keyword):
        kmeans_label_name = "_kmeans_label"
        df = data_all.copy()
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df[kmeans_label_name] = best_labels.reshape(-1, 1)
        remote_id = self.save_dataset_output(df, 'parquet')
        dataset_output = self.runtime_inputs[DATASET_INPUT]
        dataset_output.param.remote_id = remote_id
        output_columns = dataset_output.param.meta.columns
        kmeans_label_column = DatasetMetaColumn(kmeans_label_name, 'DISCRETE', 'int', 'normal', 'targetColumn')
        output_columns.append(kmeans_label_column)
        self.set_output(download_keyword, dataset_output)


def equal_frequency_sample(labels, sample_count, seed):
    """
    等频的采样
    Args:
        labels:
        sample_count:
        seed:

    Returns:
        ndarray: shape(sample_count,)

    """
    ratio = sample_count / len(labels)
    labels_count = np.bincount(labels)
    sampled_labels_count = np.ceil(labels_count * ratio)
    np.random.seed(seed)
    to_concat_list = []
    for k in range(len(sampled_labels_count)):
        k_arr = np.where(labels == k)[0]
        to_concat_list.append(np.random.choice(k_arr, int(sampled_labels_count[k])))
    ret = np.concatenate(to_concat_list)
    ret.sort()
    return ret
