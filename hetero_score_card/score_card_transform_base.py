# coding: utf-8
import time
import copy
import json
import math
import pickle
import typing
from abc import ABC
from io import StringIO
import numpy as np
import pandas as pd

from fmpc.utils.ConstUtil import ServerEnvType
from fmpc.utils.EnvUtil import EnvUtil
from fmpc.utils.FastDfsSliceUtils import save_bytes, get_bytes, get_content
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_algorithm import BaseAlgorithm
from wares.common.ware_param import ModelNodeParam, ModelWareParam, ModelMeta, DatasetMetaColumn

logger = get_fmpc_logger(__name__)

NUMERIC_TYPES = ['float', 'int', 'tinyint', 'bigint', 'double']
MODEL_INPUT = 'input1'
MODEL_INPUT2 = 'input2'
MODEL_OUTPUT = 'SCORE_MODEL'
SCORE_CARD_WARE_ID = 'cn.fudata.SCORECARD'
SCORE_CARD_REPORT = 'score_card_report'
LR_WARE_ID = 'cn.fudata.FL_LR'


class ScoreCardTransformBase(BaseAlgorithm, ABC):

    def __init__(self, ware_id, **kwargs):
        """
        :param model: 模型保存
        :param rules: woe规则
        :param base_score: 基准分
        :param base_odds: 基准ODDS
        :param pdo: 该违约概率翻倍的评分
        :param B: 评分系数
        :param A: 评分系数
        """
        super().__init__(ware_id, **kwargs)
        # model
        self.model = None
        # rules
        self.rules = None
        # settings
        self.base_score = None
        self.base_odds = None
        self.pdo = None
        self.B = None
        self.A = None

    def load_model(self):
        model_input = self.runtime_inputs[MODEL_INPUT]
        if not model_input:
            raise RuntimeError("未找到模型类型输入！")
        model_ware_id = model_input.param.ware_name
        if model_ware_id != LR_WARE_ID:
            raise TypeError(f"模型类型只能接受纵向逻辑回归类型！model_ware_id -> {model_ware_id}")
        self.model = pickle.loads(get_bytes(self.model_input.url))

    def load_predict_model(self):
        if self.job.extra.get('task_type') == "api_predict":
            model_input = self.runtime_inputs["predict_model"]
        else:
            model_input = self.runtime_inputs[MODEL_INPUT2]
        if not model_input:
            raise RuntimeError("未找到模型类型输入！")
        model_ware_id = model_input.param.ware_name
        self.model = pickle.loads(get_bytes(self.model_input.url))

    def save_model(self, model):
        """
        Args:
            # model(tuple): 模型，当有标签时，长度为4; 无标签时长度为3

        """
        model_remote_id = save_bytes(pickle.dumps(model))
        # 生成模型结果上报平台
        model_in = self.runtime_inputs[MODEL_INPUT]
        if model_in is None:
            model_output = ModelNodeParam.create_empty(self.job.currnode.node, 'NO_DATA_OWNER')
            return self.set_output(MODEL_OUTPUT, model_output)
        model_output = ModelNodeParam.create_empty(self.job.currnode.node, model_in.node_type)
        if model_in.param is not None and model_in.param.meta is not None:
            dataset_in_meta = model_in.param.meta
            model_meta = ModelMeta(dataset_in_meta.columns, dataset_in_meta.dataset_tag, dataset_in_meta.match_column,
                                   model_in.param.meta.rules, dataset_in_meta.dataset_type, None, version=self.version)
            model_output.param = ModelWareParam(model_remote_id, 'cn.fudata.SCORECARD_PREDICT', model_meta, self.job_id,
                                                self.ware.cn_name, "", self.job.inputs[0].param_name_cn)
        self.set_output(MODEL_OUTPUT, model_output)

    def update_model_report(self, report: dict):
        """
        模型报告
        Args:
            report(dict): 报告

        """
        self.ware_ctx.set_ware_output_data(SCORE_CARD_REPORT, json.dumps(report))

    def parse_settings(self, settings):
        self.base_score = settings.get('base_score')
        self.base_odds = settings.get('base_odds')
        self.pdo = settings.get('pdo')
        self.B = self.pdo / math.log(2)
        self.A = self.base_score - self.B * math.log(self.base_odds)

    def parse_common_conf(self):
        self.task_type = self.job.extra.get('task_type')

    def parse_woe_rules(self):
        model_input = self.runtime_inputs[MODEL_INPUT]
        rules = model_input.param.meta.rules
        self.validate_woe_rules(rules)
        self.rules = rules

    def validate_woe_rules(self, rules):
        # 检查本地的字段是否进行过woe转换
        feature_column_names = set(self.model_input.feature_column_names)
        if not feature_column_names:
            return
        woe_rules = set([r.column_name for r in rules if r.rule_type == 'woe'])
        without_woe_rules_set = feature_column_names - woe_rules
        if len(without_woe_rules_set) > 0:
            raise RuntimeError(f"模型有特征未进行woe转换: {without_woe_rules_set}")

    def calculate_score_weight(self):
        if len(self.model[1]) == 0 and len(self.model) == 4:
            round_ret_model = copy.deepcopy(self.model[2])
            if self.static_role() == 'GUEST':
                theta_0 = self.model[3]
                ret_theta_0 = theta_0 * self.B + self.A
                return {}, pd.Series([]), round_ret_model, ret_theta_0
            else:
                return {}, pd.Series([]), round_ret_model
        if not self.rules:
            raise IndexError("Empty rules")
        rules_dict = {}
        for rule in self.rules:
            print(rule)
            if rule.rule_type != 'woe':
                continue
            rules_dict[rule.column_name] = rule
        ret_model = {}
        round_ret_model = copy.deepcopy(self.model[2])
        theta_dict = self.model[0]
        for col in theta_dict:
            theta_w = theta_dict.get(col) * self.B
            ret_model[col] = theta_w
            round_ret_model[col] = round(theta_w, 2)
        ret_model_series = pd.Series(data=ret_model)

        if self.static_role() == 'GUEST':
            theta_0 = self.model[3]
            ret_theta_0 = theta_0 * self.B + self.A
            return ret_model, ret_model_series, round_ret_model, ret_theta_0
        else:
            return ret_model, ret_model_series, round_ret_model

    def parse_predict(self):
        """
        TODO：预测时解析数据
        :return: 解析后的字典dict
        """
        data = {}
        if self.task_type == 'batch_predict':
            result_id_dataurl = self.dataset_input.pre_dataset_remote_id
            model_content = get_content(result_id_dataurl)
            self.df_pre = pd.read_csv(StringIO(model_content), float_precision='round_trip')
            self.column_names = self.get_columns(self.model_input.columns)
            self.model_id = self.model_input.url

            # data_content = get_content(self.dataset_input.url)
            # df = pd.read_csv(StringIO(data_content), index_col=self.dataset_input.column_id)
            logger.info("开始读取原始文件: %s" % self.dataset_input.url)
            logger.info("文件类型: {}".format(self.dataset_input.content_type))
            df = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)

            logger.info("预测数据url:{}, 预测数据列名：{}, 解析的列名：{}".format(self.dataset_input.url, df.columns, self.column_names))
            for k in df.index.tolist():
                data[k] = {}
                for feature in self.column_names[1:]:
                    data[k][feature] = df.loc[k, feature]

        elif self.task_type == 'api_predict':
            self.gen_api_columns(self.model_input.columns)
            org_data = self.data_dict
            data = parse_predict_data(self.column_names, org_data)
        else:
            raise ValueError('未知的task type')
        return data

    def get_columns(self, cols: typing.List[DatasetMetaColumn]) -> typing.List[str]:
        """
        TODO：获取数据匹配特征列
        :param cols: 特征列
        :return: 匹配特征列
        """
        if cols is None or len(cols) == 0:
            return []
        else:
            column_names = self.gen_columns(cols)
            logger.info(self.dataset_input.label_name)
            logger.info(self.dataset_input.column_id)
            if self.dataset_input.has_label:
                if self.dataset_input.label_name not in column_names:
                    if self.job.ware_id != 'cn.fudata.FL_LR_PREDICT':
                        # 预测以外的算法才抛错
                        raise RuntimeError("数据集缺少目标字段：%s!" % self.dataset_input.label_name)
                else:
                    column_names.remove(self.dataset_input.label_name)
                    column_names.insert(0, self.dataset_input.label_name)
            if self.dataset_input.column_id not in column_names:
                raise RuntimeError("数据集缺少id字段：%s!" % self.dataset_input.column_id)
            else:
                column_names.remove(self.dataset_input.column_id)
                column_names.insert(0, self.dataset_input.column_id)
            return column_names

    def gen_columns(self, cols):
        """
        TODO：获取数据特征列
        :param cols: 特征列
        :return: 特征列
        """
        column_names = []
        for col in cols:
            if col.status != 'normal':
                self.log_error('ERROR: 输入特征中存在异常特征，请先使用特征筛选组件进行筛选!')
                raise RuntimeError('ERROR: 输入特征中存在异常特征，请先使用特征筛选组件进行筛选!')
            col_name = col.name
            # 数值类型判断
            if self.dataset_input.column_id != col_name and self.dataset_input.label_name != col_name \
                    and col.type not in NUMERIC_TYPES:
                logger.warn("数据列%s非数值类型！" % col_name)
                continue
            column_names.append(col_name)
        return column_names

    def gen_api_columns(self, cols: typing.List[DatasetMetaColumn]):
        """
        TODO：获取api匹配特征列
        :param cols: 特征列
        :return: 匹配特征列
        """
        if cols is None or len(cols) == 0:
            return []
        else:
            column_names = []
            if cols is not None:
                for col in cols:
                    col_name = col.name
                    if col.type not in NUMERIC_TYPES:
                        logger.warn("数据列%s非数值类型！" % col_name)
                        continue
                    column_names.append(col_name)
            self.data_dict = self.model_input.all_nodes_params[self.curr_nid].param.meta.data
            ids = list(self.data_dict.keys())
            if len(ids) == 0:
                raise ValueError("No data input!")
            data_cols = list(self.data_dict[ids[0]].keys())
            self.column_names = [val for val in column_names if val in data_cols]
            remote_id = self.model_input.url
            self.model_api = remote_id
            self.model_id_api = remote_id

    def predict_common(self, data, st, w):
        """
        TODO：Guest和host预测时解析获取数据
        :return: DataFrame数据
        """
        if self.task_type != 'api_predict':
            self.log_info('---开始运行预测函数----------')
        _t0 = time.time_ns()
        logger.info("[api_predict][predict][t0 -> %d ms]" % ((_t0 - st) // 1000000))
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = range(len(df))
        _t1 = time.time_ns()
        logger.info("[api_predict][predict][t1 -> %d ms]" % ((_t1 - st) // 1000000))
        featherlist = [x for x in df.columns]
        _t2 = time.time_ns()
        logger.info("[api_predict][predict][t2 -> %d ms]" % ((_t2 - st) // 1000000))
        df = df.T.replace('', np.nan).T
        _t3 = time.time_ns()
        logger.info("[api_predict][predict][t3 -> %d ms]" % ((_t3 - st) // 1000000))
        df = self.df_fill_nan_mode(df, featherlist, [])
        _t4 = time.time_ns()
        logger.info("[api_predict][predict][t4 -> %d ms]" % ((_t4 - st) // 1000000))
        xcs = set(df.columns)
        dws = set(w.T.columns)
        logger.info("xcs columns:{}, dws columns:{}".format(xcs, dws))
        if xcs.intersection(dws) == dws:
            df = df[w.T.columns]
        else:
            raise RuntimeError('输入数据特征有缺失')
        _t5 = time.time_ns()
        logger.info("[api_predict][predict][t5 -> %d ms]" % ((_t5 - st) // 1000000))
        return df

    def df_fill_nan_mode(self, df, all_features, cat_features):
        """
        TODO：使用众数填充缺失值
        :param df: DataFrame 数据
        :param all_features: 所有特征
        :param cat_features: 离散特征
        :return:
        """
        if self.filldict is None:
            fdicttmp = {}
            fill_nan_mode(df, all_features, cat_features, fdicttmp)
            self.filldict = fdicttmp
        else:
            for col in all_features:
                if col in self.filldict.keys():
                    df[col].fillna(self.filldict[col], inplace=True)
        return df


def parse_predict_data(column_names, org_data):
    """
    TODO：api预测时解析数据
    :return: 字典dict
    """
    data = {}
    for k in org_data:
        data[k] = {}
        for feature in column_names:
            if org_data[k][feature] == 'None':
                data[k][feature] = np.nan

            else:
                data[k][feature] = org_data[k][feature]
    return data

def fill_nan_mode(df, all_features, cat_features, fdicttmp):
    """
    TODO：众数填充缺失值
    :param df: DataFrame 数据
    :param all_features: 所有特征
    :param cat_features: 离散特征
    :param fdicttmp: 保存特征填充值
    :return:
    """
    for col in all_features:
        if col in cat_features:
            mode = df[col].mode()
            if len(mode) > 0:
                fdicttmp[col] = mode[0]
            else:
                fdicttmp[col] = 0
            df[col].fillna(fdicttmp[col], inplace=True)
        else:
            mode = df[col].round(2).mode()
            if len(mode) > 0:
                fdicttmp[col] = mode[0]
            else:
                fdicttmp[col] = 0
            df[col].fillna(fdicttmp[col], inplace=True)