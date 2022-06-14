# coding: utf-8
# ********************************************
# @Desc: 群体稳定性指标
# ********************************************

import pandas as pd
import numpy as np
import json
import copy
from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from fmpc.utils.FastDfsSliceUtils import save_content
from fmpc.utils.JsonUtil import json_serialize_job
import time

from wares.hetero_psindex.hetero_psindex_base import HeteroPSIBase
# from wares.hetero_psindex.bin_processer import BinProcesser
from wares.common.binning.binprocessing import BinProcessing

logger = get_fmpc_logger(__name__)


class HeteroPSIHost(HeteroPSIBase):
    def __init__(self, ware_id, **kwargs):
        """
        缺失值处理组件
        """
        super(HeteroPSIHost, self).__init__(ware_id, **kwargs)

    def do_ready(self):
        """ 参数准备 """
        self.log_info('======>>>>>> HOST PSIndex Ready')
        # 参数解析
        self.base_param()
        # 将job信息上传给临时变量psindex_ware_job，接口函数中会用到
        self.ware_ctx.set_ware_data("psindex_ware_job", json_serialize_job(self.job))
        self.log_info('======>>>>>> HOST PSIndex Ready Finish')

    def do_start(self):
        """ PSI处理 """
        self.log_info('======>>>>>> HOST PSIndex Start')
        try:
            # 计算特征的稳定性
            if self.feature_in_curr_node:
                logger.info('======>>>>>> 待处理的特征在host方本节点上')
                psi_json = self.cal_psindex()
                logger.info("======>>>>>> 丰富psi_json其他相关信息")
                psi_json = self.psi_json_rich(psi_json)
            # 发送psi_json
            if not self.is_owner and self.feature_in_curr_node:
                logger.info('======>>>>>> 将计算的稳定性结果发送给发起方')
                self.algo_data_transfer.psi_json.send(psi_json, self.ctx)
            # 获取psi_json
            elif self.is_owner and not self.feature_in_curr_node:
                logger.info('======>>>>>> 接受稳定性结果，本节点为发起方')
                psi_json = self.algo_data_transfer.psi_json.get(self.listener)

            logger.info("======>>>>>> 发起方汇总所有方离散特征信息，并上传离散特征信息 Host")
            self.upload_all_categorical_info()

            if self.is_owner:
                logger.info('======>>>>>> 发起方上报计算结果 host')
                logger.debug('======>>>>>> 上报结果为:{}'.format(psi_json))
                # self.ware_ctx.set_ware_data('psindex_result_data', json.dumps(psi_json))
                psi_remote_id = self.file_system_client.write_content(json.dumps(psi_json, ensure_ascii=False))
                resjson = {
                    "type": "model_psindex",
                    "job_type": "PSIndex",
                    "jobId": self.job_id,
                    "status": "success",
                    "result_remote_id": psi_remote_id,
                }
                self.ware_ctx.set_ware_data('psindex_result_data', json.dumps(resjson))
                logger.info('======>>>>>> 发起方上报结果完成')

                time.sleep(2)
                # 上报平台进入人工交互界面
                logger.info('======>>>>>> 进入人工交互界面')
                self.flow_callback(self.job, "PAUSE")
                logger.info('======>>>>>> 进入人工交互界面 结束')

        except Exception as ex:
            logger.info('======>>>>>> PSIndex计算错误 host方')
            resjson = {
                "type": "model_psindex",
                "job_type": "PSIndex",
                "jobId": self.job_id,
                "status": "failed",
                "result_remote_id": None,
            }
            self.ware_ctx.set_ware_data('psindex_result_data', json.dumps(resjson))
            # 上报平台， 组件运行失败
            self.flow_callback(self.job, "FAILED")
            raise ex
        self.log_info('======>>>>>> PSIndex Start Finish')

    def cal_psindex(self):
        """ PSI稳定性计算 """
        logger.info('======>>>>>> Host方 开始PSI稳定性计算')
        psi_json = {}

        # 获取样本特征值
        logger.info('======>>>>>> Host方 获取样本特征值')
        expected_data_df = self.expected_data_df.loc[:, [self.target_feature]]
        actual_data_df = self.actual_data_df.loc[:, [self.target_feature]]
        expected_data_df.reset_index(drop=True)
        actual_data_df.reset_index(drop=True)

        if self.bin_method == "chimerge_bin" or self.bin_method == 'distince_bin' or self.bin_method == 'custom_bin':
            # 获取guest方发送的加密的标签值
            self.log_info("---START--- host get E(y)")
            expected_label_encrypted = self.algo_data_transfer.encrypted_y.get(self.listener)
            expected_data_df = pd.concat([expected_data_df, expected_label_encrypted], axis=1)
            self.expected_data_label = expected_label_encrypted.columns[0]

        # # 获取特征（训练集,验证集）非缺失值的样本index
        # index_notna_expected = np.array((expected_data_df.loc[:, self.target_feature]).notna())
        # # 样本特征（训练集）缺失值的个数
        # num_nan_expected = np.sum(~index_notna_expected)
        # 获取非缺失值(训练集)的样本集，且为arr
        expected_data_arr = np.array(expected_data_df)
        # expected_arr_notna = expected_data_arr[index_notna_expected, :]
        # # 验证集的样本  actual_data_arr:一列feature
        actual_data_arr = np.array(actual_data_df)
        expected_data_arr_ = expected_data_arr[:, 0]
        actual_data_arr_ = actual_data_arr[:, 0]

        if len(set(expected_data_arr_)) == 1:
            raise ValueError("Excepted数据集中，所需处理特征的所有元素为同一值，目前不支持处理")

        # 初始化参数
        if self.is_iv_bin:
            bins_list = self.bins_param['bins_list_iv']
        else:
            feature_type = 1 if self.feature_type == 'continuous' else 0
            features_dict = {self.target_feature: feature_type}
            if self.bin_method == 'distince_bin':
                transfer_param = {"algo_data_transfer": self.algo_data_transfer,
                                  "listener": self.listener,
                                  "job_id": self.job_id,
                                  "ctx": self.ctx,
                                  "curr_nid": self.curr_nid}
                kw_params = {"label": self.expected_data_label, "bins": self.bins_param['num_bins'], "role": "HOST",
                             "guest_nid": self.guest_nid,
                             "data_transfer": transfer_param,
                             "send_features_dict_event_name": "features_event",
                             "get_bins_dict_event_name": "bins_event"}
                bp = BinProcessing("DISTANCE_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'frequency_bin':
                kw_params = {"q": self.bins_param['num_bins']}
                bp = BinProcessing("FREQUENCY_BIN", features_dict, expected_data_df, self.parallel, self.log_info,
                                   **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'chimerge_bin':
                transfer_param = {"algo_data_transfer": self.algo_data_transfer,
                                  "listener": self.listener,
                                  "job_id": self.job_id,
                                  "ctx": self.ctx,
                                  "curr_nid": self.curr_nid}
                kw_params = {"label": self.expected_data_label, "role": "HOST", "guest_nid": self.guest_nid,
                             "data_transfer": transfer_param,
                             "send_features_dict_event_name": "features_event",
                             "get_bins_dict_event_name": "bins_event"}
                if self.feature_type == 'continuous':
                    kw_params["con_bins"] = self.bins_param['num_bins']
                    kw_params["con_min_samples"] = self.bins_param['min_sample_num']
                    kw_params["con_threshold"] = self.bins_param['chi_threshold']
                else:
                    kw_params["cat_bins"] = self.bins_param['num_bins']
                    kw_params["cat_min_samples"] = self.bins_param['min_sample_num']
                    kw_params["cat_threshold"] = self.bins_param['chi_threshold']
                bp = BinProcessing("CHIMERGE_BIN", features_dict, expected_data_df, self.parallel, self.log_info,
                                   **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'discre_enum_bin':
                kw_params = {"label": self.expected_data_label}
                bp = BinProcessing("ENUMERATE_BIN", features_dict, expected_data_df, self.parallel, self.log_info,
                                   **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'custom_bin':
                transfer_param = {"algo_data_transfer": self.algo_data_transfer,
                                  "listener": self.listener,
                                  "job_id": self.job_id,
                                  "ctx": self.ctx,
                                  "curr_nid": self.curr_nid}
                kw_params = {"label": self.expected_data_label, "role": "HOST", "guest_nid": self.guest_nid,
                             "data_transfer": transfer_param,
                             "send_features_dict_event_name": "features_event",
                             "get_bins_dict_event_name": "bins_event"}
                if self.feature_type == 'continuous':
                    kw_params["con_param"] = self.bins_param['bins_list_custom']
                    kw_params["con_min_samples"] = self.bins_param['min_sample_num']
                else:
                    kw_params["cat_param"] = self.bins_param['bins_list_custom']
                    kw_params["cat_min_samples"] = self.bins_param['min_sample_num']
                bp = BinProcessing("CUSTOM_BIN", features_dict, expected_data_df, self.parallel, self.log_info,
                                   **kw_params)
                bins_dict = bp.get_bins_dict()
            bins_list = bins_dict[self.target_feature]

        if not isinstance(bins_list, list):
            raise ValueError(bins_list.get('msg'))

        logger.debug('======>>>>>> Host 计算出的bins_list为:{}'.format(bins_list))
        logger.info('======>>>>>> Host方 计算psi值')
        psi_sum, psi_value = self.get_psi_value(actual_data_arr_, expected_data_arr_, bins_list, self.feature_type)

        psi_json['psi'] = psi_sum
        psi_json['psiList'] = psi_value
        logger.debug('======>>>>>> Host方 计算psi值为:{}'.format(psi_sum))
        logger.debug('======>>>>>> Host方 计算psi详细信息为:{}'.format(psi_value))
        return psi_json

    def static_role(self) -> str:
        return Role.HOST
