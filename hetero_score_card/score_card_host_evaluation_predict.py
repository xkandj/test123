import json
import time
import typing
import traceback
from io import StringIO
import numpy as np
import pandas as pd
from pandas import RangeIndex
from fmpc.utils.LogUtils import get_fmpc_logger
from .score_card_transform_base import ScoreCardTransformBase
from fmpc.utils.report.ReportPrediction import PredictionService
from fmpc.utils.EnvUtil import EnvUtil
from fmpc.utils.ConstUtil import ServerEnvType
from fmpc.utils.FastDfsSliceUtils import get_content
from wares.common.ware_param import ReportWareParam
from wares.hetero_score_card.score_card_transform_base import SCORE_CARD_REPORT
from fmpc.fl.consts.fmpc_enums import ModelReportType


logger = get_fmpc_logger(__name__)


class ScoreCardTransformHostPredict(ScoreCardTransformBase):

    def __init__(self, ware_id, **kwargs):
        """
        :param ypred_result: 类型：字典，预测结果记录
        """
        super().__init__(ware_id, **kwargs)
        self.ypred_result = {}

    def do_ready(self) -> None:
        # 检查model 参数， 校验iv woe，获取model里面的rules, 获取woe分箱值
        # self.parse_woe_rules()
        # 获取model
        self.load_predict_model()
        self.parseconf()
        dw, self.wb, self.filldict = self.model['lr_model']
        self.wb = pd.DataFrame.from_dict(dw, orient='index')
        self.ret_model, self.ret_model_series, self.round_ret_model = self.model["score_model"]

    def parseconf(self):
        self.parse_common_conf()
        self.parse_common()

    def parse_common(self):
        """
        TODO 解析数据
        :return:
        """
        if self.job.extra.get('task_type') != "api_predict":
            data_input = self.dataset_input.all_nodes_params[self.curr_nid]
            if data_input.node_type != "NO_DATA_OWNER" and data_input.param is not None \
                    and self.dataset_input.type == "X":
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
            logger.info('--------------------< START Host score transfor Model Evaluate >--------------------')
            self.model_evaluating_and_save()
            self.report_result()
            logger.info(
                '---score transfor evaluate END Host, 总耗时{}ms, jobId={}'.format((time.time() - star_time) * 1000,
                                                                                 self.job_id))
            self.log_info('---score transfor END Host, 总耗时{}ms'.format((time.time() - star_time) * 1000))
        else:# 模型预测功能
            star_time = time.time()
            logger.info("*************************** START predict *******************")
            uids = list(self.data.keys())
            try:
                # model predict
                logger.info('--------------------< START HOST lr Model Predict >--------------------')
                self.predict()
                # 上报信息
                self.model_output()
            except Exception as ex:
                logger.error('--※※- Model Predict exception err={}-※※--'.format(str(ex)))
                uid = uids[0]  # 用第一个uid表示该批次预测，存在预测错误的情况
                self.ypred_result[uid] = {}
                self.ypred_result[uid]['fed_code'] = 50001
                self.ypred_result[uid]['fed_message'] = str(ex)
                self.ypred_result[uid]['data'] = {}
                self.ypred_result[uid]['data']['ypred'] = -1
                self.ypred_result[uid]['data']['ypred_prob'] = -1
                PredictionService.report_result(self.job_id, self.ypred_result)
                raise
            logger.info("..................... FINISH Host score transfer Model Predict ......................")
            logger.info(
                '---score transfor predict END Host, 总耗时{}ms, jobId={}'.format((time.time() - star_time) * 1000,
                                                                                self.job_id))
            self.log_info('---score transfor END Host, 总耗时{}ms'.format((time.time() - star_time) * 1000))


    def static_role(self) -> str:
        return 'HOST'

    def model_evaluating_and_save(self):
        """
        TODO: Model Test and model report
        return:
        """
        try:
            logger.info("--------------------<  START lr Model Test >--------------------")
            self.log_info('☆☆-----调用预测方法----')
            self.log_info('--* START lr Model Test *--')

            y_pred, y_pred_prob, yt = self.eval_predict()

            # 上报预测结果
            if not self.is_owner:
                logger.info("==当host前节点是-参与方==")
            else:
                predict_res_json_data = self.algo_data_transfer.owner_info_json_evaluate.get(self.listener)
                # 内连接
                pred_host = self.df_pre[self.dataset_input.match_column_list]
                join_data = pd.merge(pred_host, predict_res_json_data, how='inner', on=self.final_id)

                predict_remote_id = self.file_system_client.write_content(join_data.to_csv(index=False))
                logger.info("======= predict_remote_id :{} ".format(predict_remote_id))

                predict_res_json = {
                    "name": "预测结果",
                    "prefastdfs": predict_remote_id
                }
                # 上报结果
                self.ware_ctx.set_ware_output_data(key="REPORT_CSV_PRED", data=json.dumps(predict_res_json))

            self.log_info('--* FINISH lr Model Test *--')
            logger.info(".............................. FINISH lr Model Test ................................")
        except Exception as ex:
            self.log_error('---- 预测遇到错误 ----' + traceback.format_exc())
            logger.info(ex)
            raise RuntimeError(">>>>>>>>>>>Model Evaluate Error<<<<<<<<<<<<<<<<<<<<")

    def eval_predict(self, test=True):
        """
        TODO: 预测函数
        return: 预测结果
            y_pred：概率值 （0，1）之间，小数点保留
            y_pred_prob： 概率值 （0，1）之间
            yt：真实值
        """
        self.log_info('---开始运行预测函数----------')
        y_pred = None
        y_pred_prob = None
        yt = None
        self.log_info('------run_eval_predict_host----')
        # logger.info("开始读取原始数据文件pre_dataset_remote_id: %s" % self.dataset_input.pre_dataset_remote_id)
        # ori_df = pd.read_csv(StringIO(get_content(self.dataset_input.pre_dataset_remote_id)),
        #                      float_precision='round_trip')
        logger.info("开始读取评估数据文件url: %s" % self.dataset_input.url)
        logger.info("文件类型: {}".format(self.dataset_input.content_type))
        df = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)
        dfb = df[self.column_names]
        xb = self.sortColumns(dfb, test)
        featherlist = [x for x in xb.columns if x not in [self.dataset_input.column_id]]  # , self.datalabel]]
        xb = self.df_fill_nan_mode(xb, featherlist, [])
        lenb = xb.shape[1]
        xb = xb.iloc[:, 1:lenb]
        lenb = xb.shape[1]
        # xb.columns = RangeIndex(start=0, stop=lenb, step=1)

        logger.info('<<<<<<<<<<<<<<<<<< Xb.shape={}'.format(xb.shape))
        logger.info('<<<<<<<<<<<<<<<<<< wb.shape={}'.format(self.wb.shape))

        prob_score = xb.apply(lambda x: self.CardScore(x), axis=1)
        self.algo_data_transfer.probscore.send(prob_score, self.ctx, "predict_score", "test", self.curr_nid,
                                               self.curr_serial_id,
                                               async_send=True)

        xbup = xb.dot(self.wb)
        if test:
            self.algo_data_transfer.xbup.send(xbup, self.ctx, "predict", "test", self.curr_nid, self.curr_serial_id, async_send=True)
        else:
            self.algo_data_transfer.xbup.send(xbup, self.ctx, "predict", "test_false", self.curr_nid, self.curr_serial_id, async_send=True)
        return y_pred, y_pred_prob, yt

    def predict(self):
        st = time.time_ns()
        xb = self.predict_common(self.data, st, self.wb)
        # logger.info("开始读取原始数据文件pre_dataset_remote_id: %s" % self.dataset_input.pre_dataset_remote_id)
        # ori_df = pd.read_csv(StringIO(get_content(self.dataset_input.pre_dataset_remote_id)),
        #                      float_precision='round_trip')
        prob_score = xb.apply(lambda x: self.CardScore(x), axis=1)
        self.algo_data_transfer.probscore.send(prob_score, self.ctx, "predict_score", "test", self.curr_nid, self.curr_serial_id,
                                          async_send=True)

        xbup = xb.dot(self.wb)
        self.algo_data_transfer.xbup.send(xbup, self.ctx, "predict", "test", self.curr_nid, self.curr_serial_id, async_send=True)
        _t6 = time.time_ns()
        logger.info("[api_predict][predict][t6 -> %d ms]" % ((_t6 - st) // 1000000))

    def report_result(self):
        # 上报信息
        if not self.is_owner:
            logger.info("============= 当host前节点是-参与方 =============")
        else:
            logger.info("============= 当前host节点是-发起方 =============")
            # get report
            modeljson = self.algo_data_transfer.owner_info_json.get(self.listener)
            # 上报结果
            modeljson = ReportWareParam("模型报告", modeljson['assessment'],
                                        model_report_type=ModelReportType.SUPERVISED_REGRESSION_SCORECARD.value)
            self.set_output(SCORE_CARD_REPORT, modeljson)
            logger.info("===>>>" * 3 + "报告上报成功")

        logger.info(".............................. [LR] END HOST................................")

    def model_output(self):
        if not self.is_owner:
            logger.info("==当host前节点是-参与方==")
        else:
            logger.info("==当前host节点是-发起方==")
            # get report
            if self.task_type == 'batch_predict':
                y_pred_guest = self.algo_data_transfer.owner_info_json_predict.get(self.listener)
                logger.info("======= 批量下载 前10行 ：{}".format(y_pred_guest.head(10)))
                # 内连接
                pred_host = self.df_pre[self.dataset_input.match_column_list]
                join_data = pd.merge(pred_host, y_pred_guest, how='inner',
                                     on=self.dataset_input.match_columns.get('final'))

                predict_remote_id = self.file_system_client.write_content(join_data.to_csv(index=False))
                logger.info("======= predict_remote_id :{} ".format(predict_remote_id))

                predict_res_json = {
                    "name": "批量预测结果",
                    "prefastdfs": predict_remote_id
                }
                # 上报结果
                self.ware_ctx.set_ware_output_data(key="PREDICT_DOWNLOAD", data=json.dumps(predict_res_json))
                logger.info("===>>>" * 3 + "报告上报成功")

    def sortColumns(self, dfb: pd.DataFrame, test=False):
        column_id = self.dataset_input.column_id

        ta = list()
        ta.append(column_id)
        for st in dfb.columns.values:
            if st != column_id:
                ta.append(st)

        dfb = dfb[ta]
        return dfb

    def CardScore(self, x):
        # if not self.rules:
        #     raise IndexError("Empty rules")
        report = self.model["report"]
        # intercept_score = report["intercept_score"]
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
                        except Exception as ex:
                            feature = 'NAN'
                            logger.warning("特征名：{}没有匹配到NAN特征".format(key))
                        score = [table['scores'][i] for i, bins in enumerate(table['bins']) if feature == bins]
                    else:
                        raise ValueError("特征为无效的类型")
            if score is None:
                raise ValueError("特征列不匹配")
            feature_score.extend(score)
        return sum(feature_score)
