# coding: utf-8
# ********************************************
# @Desc:
# ********************************************
import copy
from abc import ABC
import pandas as pd
import json
import numpy as np

from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_algorithm import BaseAlgorithm
from fmpc.utils.FastDfsSliceUtils import get_content

logger = get_fmpc_logger(__name__)


class HeteroPSIBase(BaseAlgorithm, ABC):
    def __init__(self, ware_id, **kwargs):
        """
        num_flnodes: 合作方的个数（除无数据方节点）
        flnodes_nid: 合作方节点（除无数据方节点）
        target_feature: 待处理的目标特征/标签值
        feature_type: 特征类型, {'continuous','categorical'}
        bin_method: 分箱类型
        bins_param: 分箱信息的相关参数
        is_label: 处理的特征是否为预测的标签值
        feature_nid: 处理的特征所在nid
        feature_in_curr_node: 特征是否在当前节点
        is_iv_bin: 初始启动组件是否含有分箱信息（从iv/woe组件得到的分箱信息）
        enc_length: 密钥长度
        data_set_feature_values: 所有合作方节点的特征信息
        owner_nid:  发起方节点
        categorical_feature_enums : 当前节点的离散特征元素
        expected_data_df: 期望数据集
        expected_feature_names: 期望数据集特征名
        expected_categorical_feature_name: 期望数据集离散特征名
        expected_data_label: 期望数据集的标签名
        actual_data_df: 实际数据集
        actual_feature_names: 实际数据集特征名
        actual_categorical_feature_name: 实际数据集离散特征名
        actual_data_label: 实际数据集的标签名
        parallel: 是否并行
        guest_nid：guest方的nid
        """
        super(HeteroPSIBase, self).__init__(ware_id, **kwargs)
        self.num_flnodes = 0
        self.flnodes_nid = []
        self.target_feature = None
        self.feature_type = 'continuous'
        self.bin_method = None
        self.bins_param = {}
        self.is_label = False
        self.feature_nid = None
        self.feature_in_curr_node = False
        self.is_iv_bin = False
        self.enc_length = 1024
        self.data_set_feature_values = {}
        self.owner_nid = None
        self.categorical_feature_enums = {}
        self.expected_data_df = None
        self.expected_feature_names = None
        self.expected_categorical_feature_name = None
        self.expected_data_label = None
        self.actual_data_df = None
        self.actual_feature_names = None
        self.actual_categorical_feature_name = None
        self.actual_data_label = None
        self.parallel = False
        self.guest_nid = None
        self.has_iv_bins_list_dict = None

    def flnodes_info_param(self):
        """ 获取参与方节点信息 """
        # 不包含无数据方的合作节点及不包含无数据方的合作方的个数
        flnodes_nid = [i.node.nid for i in self.job.flnodes if i.data_type is not None]
        self.flnodes_nid = flnodes_nid
        self.num_flnodes = len(flnodes_nid)

    def feature_all_nodes_param(self):
        """ 解析获取所有节点特征信息 """
        inputs_ = self.job.inputs
        for inputs_tmp in inputs_:
            if inputs_tmp.param_name == "Excepted":
                params_ = inputs_tmp.params
                nodes_ = params_.get('nodes')
                data_set_feature_values = []
                for node_tmp in nodes_:
                    node_name = node_tmp['node']['nodeName']
                    nid = node_tmp['node']['nid']
                    if node_tmp['nodeType'] == 'OWNER' or node_tmp['nodeType'] == 'NO_DATA_OWNER':
                        self.owner_nid = nid
                    feature_data = []
                    if not (node_tmp['nodeType'] == 'NO_DATA_OWNER'):
                        dataset_type = node_tmp['dataset']['meta']['datasetType']
                        if dataset_type == 'Y':
                            self.guest_nid = nid
                        columns_ = node_tmp['dataset']['meta']['columns']
                        for columns_tmp in columns_:
                            columns_type = columns_tmp.get('columnType')
                            if columns_type is None:
                                type_ = 1 if columns_tmp['distribution'] == 'CONTINUOUS' else 2
                                each_feature = {'name': columns_tmp['name'], 'type': type_}
                                feature_data.append(each_feature)
                    each_ = {'nid': nid, 'nodeName': node_name, 'data': feature_data}
                    data_set_feature_values.append(each_)
        self.data_set_feature_values = data_set_feature_values

    def dataset_info_param(self):
        """ 解析样本数据并校验 """
        # 期望数据集input
        excepted_input = self.input_vo_dict.get('Excepted')
        # 实际数据集input
        actual_input = self.input_vo_dict.get('Actual')
        # 检查特征整列是否为空 期望样本集包含id，y，unique_idnex_unique,feature
        self.expected_data_df = self.dataset_loader.load(content_type=excepted_input.content_type,
                                                         remote_id=excepted_input.url,
                                                         columns=excepted_input.column_names)
        # 样本集里面包含的特征名称
        self.expected_feature_names = excepted_input.feature_column_names
        self.expected_categorical_feature_name = excepted_input.categorical_feature_names
        # 检查特征整列是否为空 期望样本集包含id，y，unique_idnex_unique,feature
        self.actual_data_df = self.dataset_loader.load(content_type=actual_input.content_type,
                                                       remote_id=actual_input.url,
                                                       columns=actual_input.column_names)
        # 样本集里面包含的特征名称
        self.actual_feature_names = actual_input.feature_column_names
        self.actual_categorical_feature_name = actual_input.categorical_feature_names
        # guest方获取标签
        if self.curr_role == 'GUEST':
            # 期望数据的标签名称 训练样本集的标签名称
            self.expected_data_label = excepted_input.label_name
            # 实际数据的标签名称 训练样本集的标签名称
            self.actual_data_label = actual_input.label_name
        # 两个不同数据集特征的校验
        if not (set(self.expected_feature_names).issubset(set(self.actual_feature_names))):
            raise TypeError("请确认输入的两个数据集的特征,actual数据集中特征应包含expected数据集中的特征！")
        if not (set(self.expected_categorical_feature_name).issubset(set(self.actual_categorical_feature_name))):
            raise TypeError("请确认输入的两个数据集的离散特征，actual数据集中离散特征应包含expected数据集中的离散特征！")

    def bins_info_base_param_start(self):
        """ 解析分箱基本信息  首次启动组件获得的分箱信息 """
        logger.info("======>>>>>> 首次启动组件获得的分箱信息 ")
        inputs = self.job.inputs
        logger.info("======>>>>>> 获取iv分箱信息")
        feature_iv_bins_dict = {}
        target_feature_info_iv = {'bins_info_in_currnid': False}
        target_feature_info_to_guest_iv = target_feature_info_iv
        for input_tmp in inputs:
            if input_tmp.param_name == 'Excepted':
                nodes_info = input_tmp.params.get('nodes')
                for each_node_info in nodes_info:
                    if each_node_info.get('node').get('nid') == self.curr_nid:
                        summary_id = each_node_info.get('dataset').get('summary')
                        if summary_id is None:
                            break
                        if isinstance(summary_id, dict):
                            summary_dict = summary_id
                        else:
                            summary = get_content(summary_id)
                            summary_dict = json.loads(summary)
                        if summary_dict is not None:
                            if len(summary_dict) == 0:
                                break
                            for feature_each, feature_dict in summary_dict.items():
                                bin_info_list = feature_dict.get('bin_dict')
                                if bin_info_list is not None and len(bin_info_list) > 0 and isinstance(bin_info_list, list):
                                    if target_feature_info_iv.get('feature_name') is None:
                                        target_feature_info_iv = {'feature_name': feature_each,
                                                                  'feature_nid': self.curr_nid,
                                                                  'feature_type': feature_dict.get('type'),
                                                                  'bins_list_iv': bin_info_list,
                                                                  'bins_info_in_currnid': True}
                                        target_feature_info_to_guest_iv = {'feature_name': feature_each,
                                                                           'feature_nid': self.curr_nid,
                                                                           'feature_type': feature_dict.get('type'),
                                                                           'bins_info_in_currnid': True}
                                    feature_iv_bins_dict[feature_each] = bin_info_list

        logger.info("======>>>>>> 获取默认分箱信息")
        if len(self.expected_feature_names) == 0:
            feature_one_name_type = {}
        else:
            feature_one_name = None
            feature_one_type = None
            for feature_one_name_ in self.expected_feature_names:
                data_ = np.array(self.expected_data_df.loc[:, feature_one_name_])
                if feature_one_name_ in self.expected_categorical_feature_name:
                    if not self.categorical_feature_enums[feature_one_name_].get('num_enums_big') and len(list(set(data_))) > 1:
                        feature_one_name = feature_one_name_
                        feature_one_type = 'categorical'
                else:
                    if len(list(set(data_))) > 1:
                        feature_one_name = feature_one_name_
                        feature_one_type = 'continuous'
                if feature_one_name is not None:
                    break

            # feature_one_name = self.expected_feature_names[0]
            # if feature_one_name in self.expected_categorical_feature_name:
            #     feature_one_type = 'categorical'
            # else:
            #     feature_one_type = 'continuous'
            feature_one_name_type = {'feature_name': feature_one_name, 'feature_type': feature_one_type,
                                     'feature_nid': self.curr_nid}

        feature_info = {'target_feature_info_iv': target_feature_info_to_guest_iv,
                        'feature_one_name_type': feature_one_name_type}

        logger.info("======>>>>>> 确定目标特征及分箱信息")
        if self.curr_role == 'GUEST':
            logger.info("======>>>>>> guest方汇总所有方的分箱信息")
            feature_info_list = self.algo_data_transfer.target_feature_info_to_guest.get_all(
                self.listener, [i for i in self.flnodes_nid])
            iv_bin_ = target_feature_info_to_guest_iv.get('bins_info_in_currnid')
            target_feature_result = {}
            if iv_bin_:
                target_feature_result = target_feature_info_to_guest_iv
            else:
                for each_nid, feature_info_ in enumerate(feature_info_list):
                    each_nid_feature_info_iv = feature_info_.get('target_feature_info_iv')
                    if each_nid_feature_info_iv.get('bins_info_in_currnid'):
                        target_feature_result = each_nid_feature_info_iv
                        break
            if len(target_feature_result) == 0:
                if len(feature_one_name_type) == 0 or feature_one_name_type.get('feature_name') is None:
                    for each_nid, feature_info_ in enumerate(feature_info_list):
                        each_nid_feature_one_type = feature_info_.get('feature_one_name_type')
                        if each_nid_feature_one_type.get('feature_name') is not None:
                            target_feature_result = each_nid_feature_one_type
                            break
                else:
                    target_feature_result = feature_one_name_type
            if len(target_feature_result) == 0:
                raise ValueError("所有特征均为离散特征且特征元素均大于700个或者连续特征为同一个值，暂不支持计算")
            logger.info("======>>>>>> guest方发送确认的目标特征及分箱信息")
            self.algo_data_transfer.target_feature_info.send(target_feature_result, self.ctx)
        elif self.curr_role == 'HOST':
            logger.info("======>>>>>> host方将本方的分箱信息发送给guest方")
            self.algo_data_transfer.target_feature_info_to_guest.send(feature_info, self.ctx, self.curr_nid)
            logger.info("======>>>>>> host方获取最终的目标特征及分箱信息")
            target_feature_result = self.algo_data_transfer.target_feature_info.get(self.listener)

        is_iv_ = target_feature_result.get('bins_info_in_currnid')
        self.target_feature = target_feature_result.get('feature_name')
        self.feature_nid = target_feature_result.get('feature_nid')
        self.feature_type = target_feature_result.get('feature_type')
        if is_iv_ is not None:  # iv分箱
            self.is_iv_bin = True
            if self.feature_nid == self.curr_nid:
                self.bins_param = {'bins_list_iv': target_feature_info_iv.get('bins_list_iv')}
        else:  # 默认分箱
            self.is_iv_bin = False
            # if self.feature_nid == self.curr_nid:
            if self.feature_type == 'categorical':
                self.bin_method = 'discre_enum_bin'
            else:
                self.bin_method = 'distince_bin'
                self.bins_param = {'num_bins': 10, 'min_sample_num': 100}
        return feature_iv_bins_dict

    def feature_iv_bin_status(self, feature_iv_bins_dict):
        """ 汇总所有参与方含有iv分箱信息的特征，并上报平台 """
        features_iv_bin_list_ = {}
        if self.is_iv_bin:
            features_iv_bin_list = list(feature_iv_bins_dict.keys())
            if self.curr_role == "GUEST":
                features_iv_bin_list_ = {self.curr_nid:features_iv_bin_list}
                for nid_co in self.flnodes_nid:
                    feature_list_ = self.algo_data_transfer.iv_bin_in_currnid_to_guest.get(self.listener, nid_co)
                    # features_iv_bin_list.extend(feature_list_)
                    features_iv_bin_list_[nid_co] = feature_list_
                self.algo_data_transfer.iv_bin_features.send(features_iv_bin_list_, self.ctx)

            elif self.curr_role == "HOST":
                self.algo_data_transfer.iv_bin_in_currnid_to_guest.send(features_iv_bin_list, self.ctx, self.curr_nid)
                features_iv_bin_list_ = self.algo_data_transfer.iv_bin_features.get(self.listener)
        else:
            features_iv_bin_list_ = {self.curr_nid: []}
            for nid_co in self.flnodes_nid:
                features_iv_bin_list_[nid_co] = []
        feature_iv_result = {'feature_bins_currnid':feature_iv_bins_dict,'features_list':features_iv_bin_list_}
        self.has_iv_bins_list_dict = features_iv_bin_list_
        logger.debug("======>>>>>> 特征的iv分箱信息:{}".format(feature_iv_result))
        self.ware_ctx.set_ware_data('feature_iv_result', json.dumps(feature_iv_result))

    def bins_info_base_param_restart(self, settings):
        """ 解析分箱基本信息 settings 修改PSI重新启动组件运行获得的分箱参数信息 """
        bins_params_change = settings.get('psi_params')
        # 分箱方式
        self.target_feature = bins_params_change.get('featureName')
        self.feature_nid = bins_params_change.get('nid')
        # 特征类型 1 连续性,2离散型
        feature_type = bins_params_change.get('featureType')
        bin_method_ = bins_params_change.get('binId')
        if feature_type == 1:
            self.feature_type = 'continuous'
            if bin_method_ not in ['distinceBin','frequencyBin','chiSquareBin','customBin','defaultBin']:
                raise TypeError('请确保连续特征的分箱类型是{等距分箱，等频分箱，卡方分箱，手动分箱，初始分箱}中的一种')
        elif feature_type == 2:
            self.feature_type = 'categorical'
            if bin_method_ not in ['chiSquareBin', 'customDispersedBin','enumBin','defaultBin']:
                raise TypeError('请确保连续特征的分箱类型是{卡方分箱，手动分箱，枚举分箱，初始分箱}中的一种')
        else:
            raise ValueError('请确保当个特征处理的类型是{1,2}中的一个')

        feature_iv_result = json.loads(self.ware_ctx.get_ware_data('feature_iv_result'))
        self.has_iv_bins_list_dict = feature_iv_result.get('features_list')
        logger.debug("======>>>>>> 初始分箱信息:{}".format(feature_iv_result))

        if bin_method_ == 'distinceBin':
            # 等距分箱 连续
            self.bin_method = 'distince_bin'
            bins_params_ = bins_params_change.get('binParam')
            self.bins_param = {'num_bins': bins_params_.get('subRange'),
                               'min_sample_num': bins_params_.get('minSampleNum')}
        elif bin_method_ == 'frequencyBin':
            # 等频分箱 连续
            self.bin_method = 'frequency_bin'
            bins_params_ = bins_params_change.get('binParam')
            self.bins_param = {'num_bins': bins_params_.get('subLength')}
        elif bin_method_ == 'chiSquareBin':
            # 卡方分箱 连续/离散
            self.bin_method = 'chimerge_bin'
            bins_params_ = bins_params_change.get('binParam')
            self.bins_param = {'num_bins': bins_params_.get('binNum'),
                               'min_sample_num': bins_params_.get('minSampleNum'),
                               'chi_threshold': bins_params_.get('threshold')}
        elif bin_method_ == 'customBin':
            # 连续型手动分箱
            self.bin_method = 'custom_bin'
            bins_params_ = bins_params_change.get('binParam')
            bins_list_custom_str = bins_params_.get('userDefineParam')
            # bins_list_custom = [float(i) for i in bins_list_custom_str.split(",")]
            # bins_list_custom_ = copy.deepcopy(bins_list_custom)
            # bins_list_custom_.sort()
            # if not (bins_list_custom == bins_list_custom_):
            #     raise ValueError("请确保输入的连续型特征手动分箱的格式正确，且按照从小到大一次输入")
            self.bins_param = {'bins_list_custom': bins_list_custom_str,
                               'min_sample_num': bins_params_.get('minSampleNum')}
        elif bin_method_ == 'customDispersedBin':
            # 离散特征的手动分箱
            self.bin_method = 'custom_bin'
            bins_params_ = bins_params_change.get('binParam')
            self.bins_param = {'bins_list_custom': bins_params_.get('discreteDefineParam'),
                               'min_sample_num': bins_params_.get('minSampleNum')}
        elif bin_method_ == 'enumBin':
            # 枚举分箱  离散特征
            self.bin_method = 'discre_enum_bin'
        elif bin_method_ == 'defaultBin':
            # 初始分箱
            # feature_iv_result = json.loads(self.ware_ctx.get_ware_data('feature_iv_result'))
            # self.has_iv_bins_list_dict = feature_iv_result.get('features_list')
            logger.debug("======>>>>>> 初始分箱信息:{}".format(feature_iv_result))
            self.is_iv_bin = False
            if self.has_iv_bins_list_dict is not None and self.target_feature in self.has_iv_bins_list_dict.get(self.feature_nid):
                self.is_iv_bin = True
            if self.is_iv_bin:
                if self.feature_nid == self.curr_nid:
                    feature_bins_ = feature_iv_result.get('feature_bins_currnid')
                    self.bins_param = {'bins_list_iv': feature_bins_[self.target_feature]}
            else:
                if self.feature_type == 'categorical':
                    self.bin_method = 'discre_enum_bin'
                else:
                    self.bin_method = 'distince_bin'
                    self.bins_param = {'num_bins': 10, 'min_sample_num': 100}
        else:
            raise ValueError("获取的分箱类型非列出来的任意一种")

    def categorical_feature_enums_param(self, data, categorical_feature_name):
        """
        统计data数据集中离散特征的元素
        data：dataFrame
        categorical_feature_name：list 离散特征名称
        """
        data_enums = {}
        for feature in categorical_feature_name:
            data[feature] = data[feature].astype("object")
            data[feature].fillna('NaN', inplace=True)
            data_feature_arr = np.array(data.loc[:, feature])
            bins_tem = list(set(data_feature_arr))
            if None in bins_tem:
                bins_tem.remove(None)
                bins_tem.append('NaN')
            if len(bins_tem) > 700:
                data_enums[feature] = {'num_enums_big': True}
            else:
                data_enums[feature] = {'num_enums_big': False, 'enums': bins_tem}
        return data_enums

    def feature_all_nodes_param_add_iv_info(self):
        """ 对每个特征增加是否包含iv分箱信息 """
        data_set_feature_values_new = copy.deepcopy(self.data_set_feature_values)
        for index_ in range(len(self.data_set_feature_values)):
            each_data_set_feature = self.data_set_feature_values[index_]
            each_has_iv_bins_list = self.has_iv_bins_list_dict.get(each_data_set_feature.get('nid'))
            each_data_nid_info = each_data_set_feature.get('data')
            for index_fe in range(len(each_data_nid_info)):
                feature_name = each_data_nid_info[index_fe].get('name')
                if feature_name is not None:
                    if feature_name in each_has_iv_bins_list:
                        data_set_feature_values_new[index_]['data'][index_fe]['hasIvBins'] = 1
                    else:
                        data_set_feature_values_new[index_]['data'][index_fe]['hasIvBins'] = 0
        self.data_set_feature_values = data_set_feature_values_new

    def base_param(self):
        """ 组件的基础信息 """
        logger.info("======>>>>, curr nid={}, currnode={}".format(self.curr_nid, self.job.currnode))
        # 获取参与方节点相关信息
        self.flnodes_info_param()
        logger.info("======>>>>>> 获取特征基本信息")
        self.feature_all_nodes_param()
        logger.info("======>>>>>> 获取样本数据基本信息及校验")
        self.dataset_info_param()

        logger.info("======>>>>>> 统计本节点excepted数据集离散特征的元素")
        self.categorical_feature_enums = self.categorical_feature_enums_param(self.expected_data_df,
                                                                              self.expected_categorical_feature_name)
        logger.info("======>>>>>> 获取待处理特征及分箱相关信息")
        if self.job.settings is None:
            # 初始状态的分箱信息  首次启动组件的分箱信息 并获取所有特征iv分箱信息
            feature_iv_bins_dict = self.bins_info_base_param_start()
            self.feature_iv_bin_status(feature_iv_bins_dict)
        else:
            # 修改PSI 重新启动组件从settings中获得的分箱信息
            self.bins_info_base_param_restart(self.job.settings)
        # 待处理特征是否在当前节点
        self.feature_in_curr_node = True if self.feature_nid == self.curr_nid else False

        logger.info("======>>>>>> 将iv信息add到特征基本信息中")
        self.feature_all_nodes_param_add_iv_info()


    def check_data_is_the_sample_class(self, data, typ):
        """
        检查样本的某个特征是否为常量；
        或，检查样本的标签是否唯一
        data: array, 样本的标签y值/样本的特征
        typ: feature_name 样本特征名/样本标签名
        """
        unique_val = list(np.unique(data))
        if len(unique_val) <= 1:
            raise ValueError('ERROR:{} 列值唯一, 请检查数据集该列'.format(typ))
        return unique_val

    def check_data_label(self, data, typ=None):
        """
        检查样本的标签是否唯一，且是否为[0,1]中的一个
        data: array, 样本的标签y值
        typ: 样本标签名
        """
        # 检查样本的标签值是否唯一
        unique_val = self.check_data_is_the_sample_class(data, typ)
        for val in unique_val:
            if val not in [0, 1]:
                raise ValueError('ERROR:{} 列存在非0,1值,请检查样本列数据'.format(typ))

    def get_psi_value(self, actual_data, expected_data, bins, data_type):
        """   基本OK
        根据分箱结果计算PSI
        expected_data: array，训练样本集，基准组  期望
        actual_data: array， 验证样本集，比较组   实际
        bins: list， 分箱边界值
        data_type: 数据类型，{'continuous','categorical'}
        """
        # 统计每个分箱内样本的占比
        logger.info('======>>>>>> get_psi_value 根据分箱计算psi值')
        num_expected = len(expected_data)
        num_actual = len(actual_data)
        # 每一个分箱的样本数
        logger.info('======>>>>>> 对期望数据集统计每个分箱中的样本数')
        cnt_expected, score_range_expected = get_bins_cnt_range(expected_data, bins, data_type)
        logger.info('======>>>>>> 对实际数据集统计每个分箱中的样本数')
        cnt_actual, score_range_actual = get_bins_cnt_range(actual_data, bins, data_type)
        # 实际占比
        actual_percents = cnt_actual / num_actual
        # 期望占比
        expected_percents = cnt_expected / num_expected
        delta_percents = actual_percents - expected_percents
        actual_percents_ = copy.deepcopy(actual_percents)
        expected_percents_ = copy.deepcopy(expected_percents)
        actual_percents_[actual_percents_ == 0] = 0.001  # 也可以是为1/actual_data非缺失值的样本数
        expected_percents_[expected_percents_ == 0] = 0.001
        # 计算每个分箱的psi值
        sub_psi_array = (actual_percents_ - expected_percents_) * np.log(actual_percents_ / expected_percents_)
        # psi_sum = np.sum(sub_psi_array)

        # step4: 得到最终稳定性指标    这一部分的稳定性指标后期需要根据具体的需求来进行设计
        logger.info('======>>>>>> 得到最终稳定性指标')
        psi_value = pd.DataFrame()
        psi_value['section'] = score_range_actual  # 区间
        psi_value['actual'] = actual_percents  # actual
        psi_value['expected'] = expected_percents  # expected
        psi_value['acEx'] = delta_percents  # Ac-Ex
        psi_value['acEx'] = psi_value['acEx'].apply(lambda x: round(x, 4))  # 小数点后保留4位
        psi_value['actual'] = psi_value['actual'].apply(lambda x: round(x, 4))  # 小数点后保留4位
        psi_value['expected'] = psi_value['expected'].apply(lambda x: round(x, 4))
        psi_value['lnAcDivEx'] = psi_value.apply(
            lambda row: np.log((row['actual'] + 0.001) / (row['expected'] + 0.001)), axis=1)  # In(Ac/Ex)
        psi_value['lnAcDivEx'] = psi_value['lnAcDivEx'].apply(lambda x: round(x, 4))

        psi_value['index'] = sub_psi_array  # index
        psi_value['index'] = psi_value['index'].apply(lambda x: round(x, 6))
        psi_sum = round(np.sum(psi_value['index']), 6)

        psi_value = psi_value.to_dict(orient='records')

        return psi_sum, psi_value

    def psi_json_rich(self, psi_json):
        """ 丰富返回值 psi_json """
        if self.bin_method == 'distince_bin':
            bin_id = 'distinceBin'
            bins_param = {'subRange': self.bins_param['num_bins'],
                          'minSampleNum': self.bins_param['min_sample_num']}
        elif self.bin_method == 'frequency_bin':
            bin_id = 'frequencyBin'
            bins_param = {'subLength': self.bins_param['num_bins']}
        elif self.bin_method == 'chimerge_bin':
            bin_id = 'chiSquareBin'
            bins_param = {'binNum': self.bins_param['num_bins'],
                          'minSampleNum': self.bins_param['min_sample_num'],
                          'threshold': self.bins_param['chi_threshold']}
        elif self.bin_method == 'custom_bin':
            if self.feature_type == 'continuous':
                bin_id = 'customBin'
                # bins_list_str_ = str(self.bins_param['bins_list_custom'][1:-1])
                # bins_list_str = bins_list_str_[1:-1]
                bins_list_str = self.bins_param['bins_list_custom']
                bins_param = {'userDefineParam': bins_list_str,
                              'minSampleNum': self.bins_param['min_sample_num']}
            else:
                bin_id = 'customDispersedBin'
                bins_param = {'discreteDefineParam': self.bins_param['bins_list_custom'],
                              'minSampleNum': self.bins_param['min_sample_num']}
        elif self.bin_method == 'discre_enum_bin':
            bin_id = 'enumBin'
            bins_param = {}
        else:
            if self.is_iv_bin:
                bin_id = 'defaultBin'
            else:
                bin_id = None
            bins_param = {}

        psi_json['binId'] = bin_id
        psi_json['binParam'] = bins_param
        # 特征名称
        psi_json['featureName'] = self.target_feature
        # 特征类型 {'continuous','categorical'}
        psi_json['featureType'] = 1 if self.feature_type == 'continuous' else 2
        # feature所在的nid
        psi_json['nid'] = self.feature_nid
        psi_json['dataSetFeatureValues'] = self.data_set_feature_values
        return psi_json

    def upload_all_categorical_info(self):
        """
        发起方汇总所有方离散特征信息，并上传离散特征信息
        """
        if self.is_owner:
            categorical_feature_enums_all = {self.curr_nid: self.categorical_feature_enums}
            for nid_co in self.flnodes_nid:
                cate_enums_tmp = self.algo_data_transfer.categorical_feature_enums.get(self.listener, nid_co)
                categorical_feature_enums_all[nid_co] = cate_enums_tmp
            self.ware_ctx.set_ware_data('categorical_feature_enums', json.dumps(categorical_feature_enums_all,
                                                                                cls=NpEncoder))
        else:
            self.algo_data_transfer.categorical_feature_enums.send(self.categorical_feature_enums,
                                                                   self.ctx, self.curr_nid)


def get_bins_cnt_range(data_arr, bins, data_type):
    """     OK
    使用分箱值，统计data_arr在每个分箱中的样本数
    data_arr:  样本数据，基准组
    bins: list， 分箱边界值 连续变量为[-float('inf'), 1, 5,float('inf'),'NaN']
                          离散变量为 [['1', '2'], ['4', '5'],['NaN']]
    data_type: 数据类型，{'continuous','categorical'}
    return:
       cnt_array: 每一个分箱中对应的样本个数，与score_range_list中的分箱一一对应
       score_range_list: 分箱区间如 连续变量的['(-inf,1]','(1,5]','(5,inf)', 'NaN']
                                    离散变量为["['1', '2']", "['4', '5']", "['NaN']"]
    """
    if not isinstance(bins, list):
        raise TypeError("输入的bins_list类型不对")
    if not isinstance(data_type, str):
        raise TypeError("输入的data_type类型不对")

    bins_ = copy.deepcopy(bins)
    # 每一个分箱的样本数，及分箱的区间
    cnt_list = []
    score_range_list = []
    if data_type == 'continuous':
        bins_has_nan = False
        if bins_[-1] == 'NaN':
            bins_has_nan = True
            # num_nan_actual = np.sum(np.isnan(data_arr))
            bins_.pop()
        for index in range(1, len(bins_) - 1):
            num_bin_tmp = np.sum((data_arr <= bins_[index]) & (data_arr > bins_[index - 1]))
            cnt_list.append(num_bin_tmp)
            score_range_list.append('(' + str(bins_[index - 1]) + "," + str(bins_[index]) + "]")
        num_bin_tmp = np.sum(data_arr > bins_[len(bins_) - 2])
        score_range_list.append('(' + str(bins_[len(bins_) - 2]) + ",inf)")
        cnt_list.append(num_bin_tmp)
        if bins_has_nan:
            num_nan_actual = len(data_arr) - sum(cnt_list)
            cnt_list.append(num_nan_actual)
            score_range_list.append('NaN')
    elif data_type == 'categorical':
        bins_has_nan = False
        for each_bin in bins_:
            if each_bin == ['NaN']:
                bins_has_nan = True
                nan_list = ['', 'NaN', 'NAN']
                num_nan_actual = np.sum(np.isin(data_arr, nan_list))
            else:
                num_bin_tmp = np.sum(np.isin(data_arr, each_bin))
                cnt_list.append(num_bin_tmp)
                score_range_list.append(str(each_bin))
        if bins_has_nan:
            cnt_list.append(num_nan_actual)
            score_range_list.append(str(['NaN']))
    cnt_array = np.array(cnt_list)
    return cnt_array, score_range_list


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        else:
            return super(NpEncoder, self).default(obj)
