# coding: utf-8
# ********************************************
# @Desc: 群体稳定性指标函数（PSI）
# ********************************************
import time

import pandas as pd
import numpy as np
from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.ensemble.enc_dec import EncDec
from fmpc.utils.JsonUtil import json_serialize_job
from phe import paillier

import json

from wares.hetero_psindex.hetero_psindex_base import HeteroPSIBase
# from wares.hetero_psindex.bin_processer import BinProcesser
from wares.common.binning.binprocessing import BinProcessing

logger = get_fmpc_logger(__name__)


class HeteroPSIGuest(HeteroPSIBase):
    def __init__(self, ware_id, **kwargs):
        """
        缺失值处理组件
        """
        super(HeteroPSIGuest, self).__init__(ware_id, **kwargs)

    def cal_psindex_own(self):
        """ 当计算稳定性当特征在本方时，计算稳定性的处理逻辑 """
        psi_json = {}
        # 获取样本特征值与样本标签值
        expected_data_df = self.expected_data_df.loc[:, [self.target_feature, self.expected_data_label]]
        actual_data_df = self.actual_data_df.loc[:, [self.target_feature, self.actual_data_label]]
        expected_data_df.reset_index(drop=True)
        actual_data_df.reset_index(drop=True)
        # 检查期望样本集的标签值
        typ = "expected数据集的标签" + self.expected_data_label
        self.check_data_label(expected_data_df.loc[:, self.expected_data_label], typ=typ)
        typ = "actual数据集的标签" + self.actual_data_label
        self.check_data_label(actual_data_df.loc[:, self.actual_data_label], typ=typ)
        # # 获取特征（训练集,验证集）非缺失值的样本index
        # index_notna_expected = np.array((expected_data_df.loc[:, self.target_feature]).notna())
        # # 样本特征（训练集）缺失值的个数
        # num_nan_expected = np.sum(~index_notna_expected)
        # 获取非缺失值(训练集)的样本集，且为arr
        expected_data_arr = np.array(expected_data_df)
        # expected_arr_notna = expected_data_arr[index_notna_expected, :]
        # 验证集的样本  actual_data_arr:第一列是feature，第二列是label
        actual_data_arr = np.array(actual_data_df)
        expected_data_arr_ = expected_data_arr[:, 0]
        actual_data_arr_ = actual_data_arr[:, 0]

        if len(set(expected_data_arr_)) == 1:
            raise ValueError("Excepted数据集中，所需处理特征的所有元素为同一值，目前不支持处理")
        # 分箱操作
        if self.is_iv_bin:    # 数据集中含有分箱信息
            bins_list = self.bins_param['bins_list_iv']
        else:
            # data = expected_data_df
            feature_type = 1 if self.feature_type == 'continuous' else 0
            features_dict = {self.target_feature: feature_type}
            if self.bin_method == 'distince_bin':
                kw_params = {"label": self.expected_data_label,
                             "bins": self.bins_param['num_bins'],
                             'min_samples': self.bins_param['min_sample_num']}
                bp = BinProcessing("DISTANCE_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'frequency_bin':
                kw_params = {"label": self.expected_data_label,
                             "q": self.bins_param['num_bins']}
                bp = BinProcessing("FREQUENCY_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'chimerge_bin':
                if self.feature_type == 'continuous':
                    kw_params = {"label": self.expected_data_label,
                                 "con_bins": self.bins_param['num_bins'],
                                 "con_min_samples": self.bins_param['min_sample_num'],
                                 "con_threshold": self.bins_param['chi_threshold']}
                else:
                    kw_params = {"label": self.expected_data_label,
                                 "cat_bins": self.bins_param['num_bins'],
                                 "cat_min_samples": self.bins_param['min_sample_num'],
                                 "cat_threshold": self.bins_param['chi_threshold']}
                bp = BinProcessing("CHIMERGE_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'discre_enum_bin':
                kw_params = {"label": self.expected_data_label}
                bp = BinProcessing("ENUMERATE_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            elif self.bin_method == 'custom_bin':
                if self.feature_type == 'continuous':
                    kw_params = {"label": self.expected_data_label,
                                 "con_param": self.bins_param['bins_list_custom'],
                                 "con_min_samples": self.bins_param['min_sample_num']}
                else:
                    kw_params = {"label": self.expected_data_label,
                                 "cat_param": self.bins_param['bins_list_custom'],
                                 "cat_min_samples": self.bins_param['min_sample_num']}
                bp = BinProcessing("CUSTOM_BIN", features_dict, expected_data_df, self.parallel, self.log_info, **kw_params)
                bins_dict = bp.get_bins_dict()
            bins_list = bins_dict[self.target_feature]

        if not isinstance(bins_list, list):
            raise ValueError(bins_list.get('msg'))

        # 计算psi
        logger.debug('======>>>>>> Guest方 计算出的bins_list为:{}'.format(bins_list))
        psi_sum, psi_value = self.get_psi_value(actual_data_arr_, expected_data_arr_, bins_list, self.feature_type)

        psi_json['psi'] = psi_sum
        psi_json['psiList'] = psi_value
        logger.debug('======>>>>>> Guest方 计算psi值为:{}'.format(psi_sum))
        logger.debug('======>>>>>> Guest方 计算psi详细信息为:{}'.format(psi_value))
        return psi_json

    def cal_psindex_party(self):
        """计算稳定性的特征在合作方（host方），计算稳定性的逻辑"""
        # 检查期望样本集的标签值  与上面的if分支可以合并，需要看一下怎么合并
        # 校验样本的y值
        typ = "expected数据集的标签" + self.expected_data_label
        self.check_data_label(self.expected_data_df.loc[:, self.expected_data_label], typ=typ)
        typ = "actual数据集的标签" + self.actual_data_label
        self.check_data_label(self.actual_data_df.loc[:, self.actual_data_label], typ=typ)
        # 获取需要处理的feture是在哪方，即哪个nid  等需求明确后这一步看一下最后怎么处理
        if self.bin_method == "chimerge_bin" or self.bin_method == 'distince_bin' or self.bin_method == 'custom_bin':
            # 默认guest方的样本index与host方的样本index是对齐的  如果不对齐是需要加代码逻辑的
            y_arr = np.array(self.expected_data_df.loc[:, self.expected_data_label])
            pub, priv = paillier.generate_paillier_keypair(n_length=self.enc_length)
            expected_label_encrypted_arr = EncDec().encrypted_gradient(y_arr, self.parallel, pub)
            expected_label_encrypted = pd.DataFrame({self.expected_data_label: expected_label_encrypted_arr})
            # 将加密后的数据发送给需要处理的host的方
            self.log_info("======>>>>>> guest sent E(y)")
            self.algo_data_transfer.encrypted_y.send(expected_label_encrypted, self.ctx)
            transfer_param = {"algo_data_transfer": self.algo_data_transfer,
                              "listener": self.listener,
                              "job_id": self.job_id,
                              "ctx": self.ctx,
                              "curr_nid": self.curr_nid}

        if self.bin_method == "chimerge_bin":
            BinProcessing.chimerge_bin_assist_host(self.log_info, {self.feature_nid: self.feature_nid}, priv, transfer_param)
            if self.feature_type == 'continuous':
                con_min_samples = self.bins_param['min_sample_num']
                cat_min_samples = None
            else:
                con_min_samples = None
                cat_min_samples = self.bins_param['min_sample_num']
            BinProcessing.merging_assist_host("CHIMERGE_BIN", priv, self.feature_nid,
                                              "features_event", "bins_event", transfer_param, con_min_samples,
                                              cat_min_samples)
        elif self.bin_method == 'distince_bin':
            BinProcessing.merging_assist_host("DISTANCE_BIN", priv, self.feature_nid,
                                              "features_event", "bins_event", transfer_param, self.bins_param['min_sample_num'],
                                              None)
        elif self.bin_method == 'custom_bin':
            if self.feature_type == 'continuous':
                con_min_samples = self.bins_param['min_sample_num']
                cat_min_samples = None
            else:
                con_min_samples = None
                cat_min_samples = self.bins_param['min_sample_num']
            BinProcessing.merging_assist_host("DISTANCE_BIN", priv, self.feature_nid,
                                              "features_event", "bins_event", transfer_param,
                                              con_min_samples, cat_min_samples)


    def cal_psindex(self):
        """ PSIndex稳定性计算 """
        self.log_info('>>>>Guest方 PSI开始计算 ...')
        psi_json = {}
        if self.feature_in_curr_node:
            logger.info("======>>>>>> 计算稳定性的特征在guest方")
            psi_json = self.cal_psindex_own()
            logger.info("======>>>>>> 丰富psi_json其他相关信息")
            psi_json = self.psi_json_rich(psi_json)
        else:
            logger.info("======>>>>>> 计算稳定性的特征在host方")
            self.cal_psindex_party()
        return psi_json

    def do_ready(self):
        """ 参数准备    该方法中默认guest方与host方的样本是对齐的"""
        self.log_info('======>>>>>> PSIndex Ready')
        # 参数解析，获取组件的基本信息，包含数据相关信息，分箱相关信息及节点相关信息
        self.base_param()
        # 将job信息上传给临时变量psindex_ware_job，接口函数中会用到
        self.ware_ctx.set_ware_data("psindex_ware_job", json_serialize_job(self.job))
        self.log_info('======>>>>>> PSIndex Ready Finish')

    def do_start(self):
        """ PSI处理 """
        self.log_info('======>>>>>> PSIndex Start')
        logger.info('>>>>Guest方 PSI组件启动处理={} ...'.format(self.job.job_id))
        try:
            # 计算PSI
            logger.info('PSI: 开始PSI计算 .............')
            psi_json = self.cal_psindex()

            if self.is_owner and not self.feature_in_curr_node:
                psi_json = self.algo_data_transfer.psi_json.get(self.listener)
            elif not self.is_owner and self.feature_in_curr_node:
                self.algo_data_transfer.psi_json.send(psi_json, self.ctx)

            logger.info("======>>>>>> 发起方汇总所有方离散特征信息，并上传离散特征信息 Guest")
            self.upload_all_categorical_info()

            if self.is_owner:
                logger.info('======>>>>>> 发起方上报计算结果 guest')
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
                logger.info('======>>>>>> 完成人工交互界面')
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

    def static_role(self) -> str:
        return Role.GUEST
