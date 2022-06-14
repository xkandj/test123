import json
import math
import time
import traceback
import random
import typing
from functools import reduce
from io import StringIO
import numpy as np
import pandas as pd
from pandas import RangeIndex
from fmpc.utils.LogUtils import get_fmpc_logger
from fmpc.utils.ConstUtil import ServerEnvType
from fmpc.utils.FastDfsSliceUtils import get_content
from fmpc.utils.EnvUtil import EnvUtil
from .score_card_transform_base import ScoreCardTransformBase
from fmpc.utils.report.ReportPrediction import PredictionService
from wares.common.service.evaluate_service import BinaryClassificationEvaluate, ScoreCradEvaluate
from wares.common.ware_param import ReportWareParam
from wares.hetero_score_card.score_card_transform_base import SCORE_CARD_REPORT
from fmpc.fl.consts.fmpc_enums import ModelReportType


logger = get_fmpc_logger(__name__)


class ScoreCardTransformGuestPredict(ScoreCardTransformBase):

    def __init__(self, ware_id, **kwargs):
        """
        :param ypred_result: 类型：字典，预测结果记录
        :param sigmoid: sigmoid函数
        """
        super().__init__(ware_id, **kwargs)
        self.ypred_result = {}
        self.sigmoid = lambda x: 1. / (1 + np.exp(-x))

    def do_ready(self) -> None:
        # 检查model 参数， 校验iv woe，获取model里面的rules, 获取woe分箱值
        # self.parse_woe_rules()
        # 获取model
        self.load_predict_model()
        self.parseconf()
        dw, self.wa, self.filldict, self.intercept = self.model["lr_model"]
        self.wa = pd.DataFrame.from_dict(dw, orient='index')
        self.ret_model, self.ret_model_series, self.round_ret_model, self.ret_theta_0 = self.model["score_model"]
        self.report = self.model["report"]

    def parseconf(self):
        self.parse_common_conf()
        self.parse_common()

    def parse_common(self):
        if self.job.extra.get('task_type') != "api_predict":
            data_input = self.dataset_input.all_nodes_params[self.curr_nid]
            if data_input.node_type != "NO_DATA_OWNER" and data_input.param is not None \
                    and self.dataset_input.type == 'Y':
                if EnvUtil.get_server_env_type() != ServerEnvType.dev:
                    result_id_dataurl = self.dataset_input.pre_dataset_remote_id
                    content = get_content(result_id_dataurl)
                    self.df_pre = pd.read_csv(StringIO(content), float_precision='round_trip')
                    self.final_id = self.dataset_input.match_columns.get('final')
                # 特征解锁
                if self.task_type == "model_evaluate":
                    cols = self.model_input.columns
                    cols = [col for col in cols if self.dataset_input.ori_column_id != col.name]
                    self.column_names = self.get_columns(cols)
                else:
                    self.data = self.parse_predict()
        else:
            self.data = self.parse_predict()

    def do_start(self):

        if self.task_type == 'model_evaluate':
            star_time = time.time()
            # model evaluate
            logger.info('--------------------< START GUEST score transfor Model Evaluate >--------------------')
            y_pred, y_pred_prob, yt, y_pred_out, y_prob_score = self.model_evaluating()
            modeljson = self.model_evaluate_json(y_pred, y_pred_prob, yt, y_pred_out, y_prob_score)
            self.report_result(modeljson)
            logger.info('---score transfor evaluate END GUEST, 总耗时{}ms, jobId={}'.format((time.time() - star_time) * 1000, self.job_id))
            self.log_info('---score transfor END GUEST, 总耗时{}ms'.format((time.time() - star_time) * 1000))
        else:  # 模型预测功能
            star_time = time.time()
            logger.info("*************************** START predict *******************")
            uids = list(self.data.keys())
            try:
                # model predict
                logger.info('--------------------< START GUEST score transfor Model Predict >--------------------')
                y_pred, y_pred_prob, y_prob_score = self.predict()
            except Exception as ex:
                logger.error('--※※- Model Predict exception err={}-※※--'.format(str(ex)))
                self.ex_rollback(uids[0], str(ex))
                if self.task_type == 'batch_predict':
                    """
                    TODO:批量预测
                    """
                    pass
                else:
                    logger.info("========== 异常回调 ==========")
                    PredictionService.report_result(self.job_id, self.ypred_result)
                raise
            logger.info("..................... FINISH GUEST score transfer Model Predict ......................")
            self.model_output(y_pred, y_pred_prob, y_prob_score, uids)
            logger.info(
                '---score transfor predict END GUEST, 总耗时{}ms, jobId={}'.format((time.time() - star_time) * 1000,
                                                                                self.job_id))
            self.log_info('---score transfor END GUEST, 总耗时{}ms'.format((time.time() - star_time) * 1000))

    def model_evaluating(self):
        """
        TODO: 模型测试
        return: 测试结果
            y_pred：概率值 （0，1）之间，小数点保留
            y_pred_prob： 概率值 （0，1）之间
            yt：真实值
            y_pred_out：预测值和真实值对齐结果
        """
        try:
            # model test
            logger.info("--------------------<  START score transfor Model Evaluate >--------------------")
            self.log_info('--* START score transfor Model Test *--')
            y_pred, y_pred_prob, yt, y_pred_out, y_prob_score = self.eval_predict()

            self.log_info('--* FINISH score transfor Model Test *--')
            logger.info(".............................. FINISH score transfor Model Test ................................")
        except Exception as ex:
            logger.info(traceback.format_exc())
            raise RuntimeError(">>>>>>>>>>>Model Evaluate Error<<<<<<<<<<<<<<<<<<<<")
        return y_pred, y_pred_prob, yt, y_pred_out, y_prob_score

    def eval_predict(self, test=True):
        """
        TODO: 预测函数
        return: 预测结果
            y_pred：概率值 （0，1）之间，小数点保留
            y_pred_prob： 概率值 （0，1）之间
            yt：真实值
            y_pred_out：预测值和真实值对齐结果
        """
        y_pred = None
        y_pred_prob = None
        yt = None
        self.log_info('------run_eval_predict_guest----')
        # logger.info("开始读取原始数据文件pre_dataset_remote_id: %s" % self.dataset_input.pre_dataset_remote_id)
        # ori_df = pd.read_csv(StringIO(get_content(self.dataset_input.pre_dataset_remote_id)),
        #                       float_precision='round_trip')
        logger.info("开始读取评估数据文件url: %s" % self.dataset_input.url)
        logger.info("文件类型: {}".format(self.dataset_input.content_type))
        df = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)
        if df.columns is not None and self.column_names is not None:
            logger.info(
                "评估url:{}, 读取的数据列名：{}, 解析的列名：{}".format(self.dataset_input.url, df.columns, self.column_names))
        dfa = df[self.column_names]
        xa = self.sortColumns(dfa, test)
        featherlist = [x for x in xa.columns if x not in [self.dataset_input.column_id]]  # , self.datalabel]]
        xa = self.df_fill_nan_mode(xa, featherlist, [])
        lena = xa.shape[1]
        yt = xa.iloc[:, 1]
        xao = xa.iloc[:, 2:lena]
        y_pred_out = pd.DataFrame(xa.iloc[:, 0])
        y_pred_out.columns = ['id']
        lenao = xao.shape[1]

        y_prob_score = xao.apply(lambda x: self.CardScore(x), axis=1)
        score_list = []
        for n in self.role_parser.get_nodes('HOST'):
            host_score = self.algo_data_transfer.probscore.get(self.listener,
                                                               "predict_score", "test", n.node.nid,
                                                               n.node.serial_id)
            score_list.append(host_score)
        logger.info("============finish host score_list receive===============================")
        prob_score = reduce(add, score_list)
        y_prob_score += prob_score

        if lenao > 0:
            xaup = xao.dot(self.wa)
        else:
            xaup = 0.0
        xbup_list = []
        if test:
            for n in self.role_parser.get_nodes('HOST'):
                host_xbup = self.algo_data_transfer.xbup.get(self.listener,
                                                             "predict", "test", n.node.nid, n.node.serial_id)
                xbup_list.append(host_xbup)
        else:
            for n in self.role_parser.get_nodes('HOST'):
                host_xbup = self.algo_data_transfer.xbup.get(self.listener,
                                                             "predict", "test_false", n.node.nid, n.node.serial_id)
                xbup_list.append(host_xbup)
        logger.info("============finish host xbup_list receive===============================")
        xbup = reduce(add, xbup_list)
        xu = xaup + xbup + self.intercept
        y_pred_prob = xu.apply(sigmoid)

        y_pred = xu.apply(sigmoid).apply(np.round)
        y_pred_out = y_pred_out.assign(y_pred=y_pred_prob)
        y_pred_out = y_pred_out.assign(y=yt)
        y_pred_out = y_pred_out.assign(y_score=y_prob_score)
        return y_pred, y_pred_prob, yt, y_pred_out, y_prob_score

    def CardScore(self, x):
        # if not self.rules:
        #     raise IndexError("Empty rules")
        report = self.model["report"]
        intercept_score = report["intercept_score"]
        scores_table = report["score_table"]
        if len(x) != len(scores_table):
            raise RuntimeError("数据特征和woe特征不匹配")
        feature_score = []
        for key, value in x.items():
            score = None
            for table in scores_table:
                if table['name'] == key:
                    if table['distributeType'] == 'CONTINUOUS':
                        score = [table['scores'][i] for i, bins in enumerate(table['bins'])
                                 if float(bins.split(',')[0]) < value < float(bins.split(',')[1])]
                    elif table['distributeType'] == 'DISCRETE':
                        try:
                            feature = table['featureMap'][value]
                        except Exception as e:
                            feature = 'NAN'
                            logger.warning("特征名：{}没有匹配到NAN特征".format(key))
                        score = [table['scores'][i] for i, bins in enumerate(table['bins']) if feature == bins]
                    else:
                        raise ValueError("特征为无效的类型")
            if score is None:
                raise ValueError("特征列不匹配")
            feature_score.extend(score)
        return sum(feature_score) + intercept_score


    def predict(self):
        st = time.time_ns()
        xa = self.predict_common(self.data, st, self.wa)
        # logger.info("开始读取原始数据文件pre_dataset_remote_id: %s" % self.dataset_input.pre_dataset_remote_id)
        # ori_df = pd.read_csv(StringIO(get_content(self.dataset_input.pre_dataset_remote_id)),
        #                      float_precision='round_trip')
        y_prob_score = xa.apply(lambda x: self.CardScore(x), axis=1)
        score_list = []
        for n in self.role_parser.get_nodes('HOST'):
            host_score = self.algo_data_transfer.probscore.get(self.listener,
                                                               "predict_score", "test", n.node.nid,
                                                               n.node.serial_id)
            score_list.append(host_score)
        prob_score = reduce(add, score_list)
        y_prob_score += prob_score

        if xa.shape[1] > 0:
            xaup = xa.dot(self.wa)
        else:
            xaup = 0.0
        _t6 = time.time_ns()
        logger.info("[api_predict][predict][t6 -> %d ms]" % ((_t6 - st) // 1000000))
        xbup_list = []
        for n in self.role_parser.get_nodes('HOST'):
            host_xbup = self.algo_data_transfer.xbup.get(self.listener,
                                                         "predict", "test", n.node.nid, n.node.serial_id)
            xbup_list.append(host_xbup)
        _t7 = time.time_ns()
        logger.info("[api_predict][predict][t7 -> %d ms]" % ((_t7 - st) // 1000000))
        xbup = reduce(add, xbup_list)
        xu = xaup + xbup + self.intercept
        _t8 = time.time_ns()
        logger.info("[api_predict][predict][t8 -> %d ms]" % ((_t8 - st) // 1000000))
        y_pred_prob = xu.astype(np.float64).apply(self.sigmoid)
        _t9 = time.time_ns()
        logger.info("[api_predict][predict][t9 -> %d ms]" % ((_t9 - st) // 1000000))
        y_pred = xu.astype(np.float64).apply(self.sigmoid).apply(np.round)
        _t10 = time.time_ns()
        logger.info("[api_predict][predict][t10 -> %d ms]" % ((_t10 - st) // 1000000))
        for curr_uid in xu.index:
            self.ypred_result[curr_uid] = {}
            self.ypred_result[curr_uid]['fed_code'] = 0
            self.ypred_result[curr_uid]['fed_message'] = "success"
            self.ypred_result[curr_uid]['data'] = {}
            self.ypred_result[curr_uid]['data']['ypred'] = int(y_pred.loc[curr_uid].values[0])
            self.ypred_result[curr_uid]['data']['ypred_prob'] = y_pred_prob.loc[curr_uid].values[0]
        _t11 = time.time_ns()
        logger.info("[api_predict][predict][t11 -> %d ms]" % ((_t11 - st) // 1000000))
        return y_pred, y_pred_prob, y_prob_score

    def model_evaluate_json(self, y_pred, y_pred_prob, yt, y_pred_out, y_prob_score):
        """
        TODO: 模型评估
        param y_pred：概率值 （0，1）之间，小数点保留
        param y_pred_prob： 概率值 （0，1）之间
        param yt：真实值
        param y_pred_out：预测值和真实值对齐结果
        return : 模型
        """
        try:
            # model evaluate
            logger.info("--------------------<  START score card Model Evaluate >--------------------")
            self.log_info('--* START score card Model Evaluate *--')
            # 评估报告输出
            modeljson = ScoreCradEvaluate.score_crad_generate_report(np.array(yt.astype("int64")), y_pred,
                                             np.array(y_pred_prob).reshape(y_pred_prob.shape[0]),
                                             np.array(y_prob_score).reshape(y_prob_score.shape[0]), self.job_id)
            logger.info("========eval model report modeljson:{}".format(modeljson))
            # 生成结果json上报
            if self.is_owner:
                logger.info("批量下载 前10行 ：{}".format(y_pred_out.head(10)))
                logger.info("y_pred: {}".format(y_pred))
                y_pred = pd.DataFrame({'y_pred': np.array(y_pred).reshape(y_pred.shape[0])})
                for id in self.dataset_input.match_column_list:
                    y_pred.insert(0, id, self.df_pre[id], allow_duplicates=True)
                y_pred['y_pred_prob'] = y_pred_prob
                y_pred['y_prob_score'] = y_prob_score
                logger.info("======= 批量下载 前10行 ：{}".format(y_pred.head(10)))
                predict_remote_id_test = self.file_system_client.write_content(y_pred.to_csv(index=False))
                predict_res_json = {
                    "name": "预测结果",
                    "prefastdfs": predict_remote_id_test
                }
                self.ware_ctx.set_ware_output_data("REPORT_CSV_PRED", json.dumps(predict_res_json))
            else:
                logger.info("y_pred: {}".format(y_pred))
                y_pred = pd.DataFrame({'y_pred': np.array(y_pred).reshape(y_pred.shape[0])})
                ID_ = self.final_id
                y_pred.insert(0, ID_, self.df_pre[ID_])
                y_pred['y_pred_prob'] = y_pred_prob
                y_pred['y_prob_score'] = y_prob_score
                logger.info("======= 批量下载 前10行 ：{}".format(y_pred.head(10)))
                self.algo_data_transfer.owner_info_json_evaluate.send(y_pred, self.ctx, async_send=True)

            self.log_info('--* FINISH lr Model Evaluate *--')

            logger.info("================evaluate report to web============================")
            yt_prob_report_df = pd.DataFrame({'yt': np.array(yt.astype("int64")).reshape(yt.shape[0])})
            yt_prob_report_df['y_pred_prob'] = y_pred_prob
            yt_prob_report_df['y_prob_score'] = y_prob_score
            evaluate_remote_id_ = self.file_system_client.write_content(yt_prob_report_df.to_csv(index=False))
            self.ware_ctx.set_ware_data(key="train_test_remote_id_and_job_id",
                                        data=json.dumps([evaluate_remote_id_, self.job_id]))
            logger.info("================evaluate  report to web finish============================")

            logger.info("..................... FINISH lr Model Evaluate ......................")
        except Exception as ex:
            logger.info(traceback.format_exc())
            self.log_error('--- 评估遇到错误 ----' + traceback.format_exc())
            raise RuntimeError(">>>>>>>>>>>Model Evaluate Error<<<<<<<<<<<<<<<<<<<<")
        return modeljson

    def sortColumns(self, dfa: pd.DataFrame, test=False):
        ta = []
        tar = ''
        column_id = self.dataset_input.column_id
        ta.append(column_id)
        if test:
            if self.dataset_input.has_label:
                self.check_dataset(dfa)
                tar = self.dataset_input.label_name
                ta.append(tar)
        else:
            self.check_dataset(dfa)
            tar = self.dataset_input.label_name
            ta.append(tar)

        for st in dfa.columns.values:
            if st != column_id and st != tar:
                ta.append(st)

        dfa = dfa[ta]
        return dfa

    def check_dataset(self, df):
        if len(set(df[self.dataset_input.label_name].value_counts().index) - {0, 1}) > 0:
            self.log_error('数据集目标字段包含【0,1】之外的值')
            raise RuntimeError('数据集目标字段{}包含【0,1】之外的值'.format(self.dataset_input.label_name))
        if self.task_type != 'model_evaluate' and len(
                set(df[self.dataset_input.label_name].value_counts().index)) != 2:
            self.log_error('数据集目标字段为单一值')
            raise RuntimeError('数据集目标字段{}为单一值'.format(self.dataset_input.label_name))

    def static_role(self) -> str:
        return 'GUEST'

    def report_result(self, modeljson):
        # 上报信息
        if self.is_owner:
            logger.info("============= 当前guest节点是正常-发起方 =============")
            modeljson = ReportWareParam("模型报告", modeljson['assessment'],
                                        model_report_type=ModelReportType.SUPERVISED_REGRESSION_SCORECARD.value)
            logger.info("========当前guest节点 eval model report modeljson:{}".format(modeljson))
            self.set_output(SCORE_CARD_REPORT, modeljson)
            logger.info("===>>>" * 3 + "报告上报成功")
        else:
            logger.info("============= 当前guest节点是-参与方 =============")
            # send report to owner
            self.algo_data_transfer.owner_info_json.send(modeljson, self.ctx, async_send=True)

        logger.info(".............................. [LR] END GUEST................................")

    def model_output(self, y_pred, y_pred_prob, y_prob_score, uids):
        if self.task_type == 'batch_predict':
            # 上报信息
            if self.is_owner:
                logger.info("==当前guest节点是正常-发起方==")
                y_pred.columns = ["y_pred"]
                for column_id in self.dataset_input.match_column_list:
                    y_pred.insert(0, column_id, self.df_pre[column_id])
                y_pred['y_pred_prob'] = y_pred_prob
                y_pred['y_prob_score'] = y_prob_score
                logger.info("======= 批量下载 前10行 ：{}".format(y_pred.head(10)))
                predict_remote_id = self.file_system_client.write_content(y_pred.to_csv(index=False))
                logger.info("======= predict_remote_id :{} ".format(predict_remote_id))

                predict_res_json = {
                    "name": "批量预测结果",
                    "prefastdfs": predict_remote_id
                }
                self.ware_ctx.set_ware_output_data("PREDICT_DOWNLOAD", json.dumps(predict_res_json))
                logger.info("===>>>" * 3 + "报告上报成功")

            else:
                logger.info("==当前guest节点是-参与方 ==")
                y_pred.columns = ["y_pred"]
                ID_ = self.dataset_input.match_columns.get('final')
                y_pred.insert(0, ID_, self.df_pre[ID_])
                y_pred['y_pred_prob'] = y_pred_prob
                logger.info("======= 批量下载 前10行 ：{}".format(y_pred.head(10)))
                self.algo_data_transfer.owner_info_json_predict.send(y_pred, self.ctx, async_send=True)
        else:
            try:
                logger.info("========== 正常回调 ==========")
                PredictionService.report_result(self.job_id, self.ypred_result)
            except Exception as ex:
                logger.error('模型API预测出错了，error={}'.format(str(ex)))
                self.log_error('模型API预测出错了，error={}'.format(str(ex)))
                self.ex_rollback(uids[0], str(ex))
                logger.info("========== 异常回调 ==========")
                PredictionService.report_result(self.job.job_id, self.ypred_result)
                raise ex

    def ex_rollback(self, uid, ex):
        # 用第一个uid表示该批次预测，存在预测错误的情况
        self.ypred_result[uid] = {}
        self.ypred_result[uid]['fed_code'] = 50001
        self.ypred_result[uid]['fed_message'] = ex
        self.ypred_result[uid]['data'] = {}
        self.ypred_result[uid]['data']['ypred'] = -1
        self.ypred_result[uid]['data']['ypred_prob'] = -1


def add(x, y):
    return x + y


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def Prob2Score(prob, A, B):
    y = np.log(prob/(1-prob))
    return float(A-B*y)