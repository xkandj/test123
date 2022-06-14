import time
import json
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_score_card.score_card_transform_base import ScoreCardTransformBase, SCORE_CARD_REPORT
from wares.common.fmpc_error import RoleError
from wares.common.ware_param import ReportWareParam
from fmpc.fl.consts.fmpc_enums import ModelReportType

logger = get_fmpc_logger(__name__)


class ScoreCardTransformNodataPredict(ScoreCardTransformBase):

    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)

    def do_ready(self) -> None:
        # 解析settings
        # self.parse_settings(self.job.settings)
        self.task_type = self.job.extra.get('task_type')

    def do_start(self):
        self.log_info(f'无数据方开始评估或预测...')
        if not self.is_owner:
            raise RoleError('角色异常！无数据方必须是发起方！')
        if self.task_type == 'model_evaluate':
            star_time = int(round(time.time() * 1000))
            predict_res_json_data = self.algo_data_transfer.owner_info_json_evaluate.get(self.listener)
            logger.info('predict_res_json_data 333: {}'.format(predict_res_json_data))
            predict_remote_id = self.file_system_client.write_content(predict_res_json_data.to_csv(index=False))
            logger.info("======= predict_remote_id :{} ".format(predict_remote_id))
            predict_res_json = {
                "name": "批量预测结果",
                "prefastdfs": predict_remote_id
            }
            self.ware_ctx.set_ware_output_data("REPORT_CSV_PRED", json.dumps(predict_res_json))

            # 上报模型报告
            modeljson = self.algo_data_transfer.owner_info_json.get(self.listener)
            logger.info("============= 当前节点是无数据方 =============")
            modeljson = ReportWareParam("模型报告", modeljson['assessment'],
                                        model_report_type=ModelReportType.SUPERVISED_REGRESSION_SCORECARD.value)
            self.set_output(SCORE_CARD_REPORT, modeljson)
            end_time = int(round(time.time() * 1000))
            logger.info("..............................< Lr_eval end >.............................")
            logger.info('=======>>[All_time], 总耗时 {} 毫秒/ms'.format(end_time - star_time))
            self.log_info('--- Lr_eval nodata End ---')
            self.log_info('总耗时 {} 毫秒/ms '.format(end_time - star_time))
        else:
            if self.task_type == "batch_predict":
                star_time = int(round(time.time() * 1000))
                predict_res_json_data = self.algo_data_transfer.owner_info_json_predict.get(self.listener)
                logger.info("======= predict_res_json_data: {} ".format(predict_res_json_data))
                predict_remote_id = self.file_system_client.write_content(predict_res_json_data.to_csv(index=False))
                logger.info("======= predict_remote_id :{} ".format(predict_remote_id))
                predict_res_json = {
                    "name": "批量预测结果",
                    "prefastdfs": predict_remote_id
                }
                self.ware_ctx.set_ware_output_data("PREDICT_DOWNLOAD", json.dumps(predict_res_json))
                end_time = int(round(time.time() * 1000))
                logger.info("..............................< Lr_pre end >.............................")
                logger.info('=======>>[All_time], 总耗时 {} 毫秒/ms'.format(end_time - star_time))
                self.log_info('--- Lr_pre nodata End ---')
                self.log_info('总耗时 {} 毫秒/ms '.format(end_time - star_time))

    def static_role(self) -> str:
        return 'NODATA'
