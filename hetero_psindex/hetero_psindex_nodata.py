# encoding:utf-8
import json
import time

from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from fmpc.utils.FastDfsSliceUtils import save_content
from fmpc.utils.JsonUtil import json_serialize_job

from wares.hetero_psindex.hetero_psindex_base import HeteroPSIBase

logger = get_fmpc_logger(__name__)


class HeteroPSINodata(HeteroPSIBase):
    """
        DT guest算法：算法名+角色，job为算法对象属性
    """
    def __init__(self, ware_id, **kwargs):
        """
        创建算法对象
        """
        super(HeteroPSINodata, self).__init__(ware_id, **kwargs)

    def do_ready(self) -> None:
        self.log_info('======>>>>>> PSIndex Ready')
        # # 本节点是否为发起方  框架已经解析
        # self.is_owner = True
        # 获取参与方节点相关信息
        self.flnodes_info_param()
        # 将job信息上传给临时变量psindex_ware_job，接口函数中会用到
        self.ware_ctx.set_ware_data("psindex_ware_job", json_serialize_job(self.job))
        self.log_info('======>>>>>> PSIndex Ready Finish')

    def do_start(self) -> None:
        self.log_info('======>>>>>> Nodata PSIndex Start')
        logger.info("======>>>> 当前节点是-无数据发起方")
        try:
            logger.info("======>>>>>> 发起方汇总所有方离散特征信息，并上传离散特征信息 Nodata")
            self.upload_all_categorical_info()

            if self.is_owner:
                logger.info('======>>>> 当前节点是-无数据发起方')
                psi_json = self.algo_data_transfer.psi_json.get(self.listener)
                logger.debug('======>>>>>> 上报结果为:{}'.format(psi_json))
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
                # 上报平台进入人工交互界面
                time.sleep(2)
                logger.info('======>>>>>> 进入人工交互界面')
                self.flow_callback(self.job, "PAUSE")
                logger.info('======>>>>>> 进入人工交互界面 结束')
        except Exception as ex:
            logger.info('======>>>>>> PSIndex上报失败 nodata方')
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
        self.log_info('======>>>>>> Nodata PSIndex Start Finish')

    def static_role(self) -> str:
        return Role.NODATA
