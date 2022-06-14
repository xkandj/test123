# coding: utf-8

import json
import traceback
from flask import request, Blueprint
from fmpc.base.Node import Job
from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from fmpc.base.WareContext import WareContext
from fmpc.utils.FastDfsSliceUtils import get_content
from fmpc.utils.HttpUtil import HttpUtil
from fmpc.utils.RouterService import RouterService
from fmpc.utils.CallApiUtil import CallApiUtil
from fmpc.base.Algorithm import Context

from wares.hetero_psindex.hetero_psindex_guest import HeteroPSIGuest
from wares.hetero_psindex.hetero_psindex_host import HeteroPSIHost
from wares.hetero_psindex.hetero_psindex_nodata import HeteroPSINodata
from wares.common.job_role_parser import JobRoleParser, HeteroModelJobRoleParser
from wares.common.base_ware import BaseWare
from wares.hetero_psindex.hetero_psindex_algo_data_transfer import PSIndexAlgoDataTransfer

from fmpc.base.ResultModels import error_result_json, Result

hetero_psindex_app = Blueprint("hetero_psindex_ware_app", __name__, static_folder="h", static_url_path="/h")
logger = get_fmpc_logger(__name__)

# ware_id
WARE_ID = "cn.fudata.FL_PSINDEX"
# error missing params
ERROR_MISSING_PARAMS = "missing params"


def succ_result_json(data):
    return Result('0', 'OK', data).json_dumps()


def flow_callback(flow_id, container_id, ware_id, ware_version, job_id,
                  status):
    return CallApiUtil.flow_callback(flow_id, container_id, ware_id,
                                     ware_version, job_id, status)


@hetero_psindex_app.route("/analysis/psi/getResult", methods=["POST"])
def get_result():
    """ 修改PSI 对应的按钮"""
    try:
        logger.info("==>> 查询特征变量分箱:[/feature/query]...")
        req_data = request.data
        if req_data is None:
            return error_result_json('100000', ERROR_MISSING_PARAMS)
        params = json.loads(req_data)
        logger.debug("==>> get_result params:{}".format(params))
        ware_id = params.get("wareId")
        flow_id = params.get("flowId")
        cid = params.get("containerId")
        check_id(flow_id, cid)
        logger.debug('flowId：{}, wareId：{}'.format(flow_id, ware_id))
    except Exception as ex:
        return error_result_json("100001", f"参数获取异常[flowId/containerId].{str(ex)}")

    try:
        ware_id = WARE_ID
        logger.debug(f'flowId：{flow_id}, containerId：{cid}, wareId：{ware_id}')
        ctx = WareContext(flow_id, ware_id, cid)
        logger.debug("======>>>>>> query_result_ctx:{}".format(ctx))
        # 获取psi的结果数据
        resjson = json.loads(ctx.get_ware_data('psindex_result_data'))
        remote_id = resjson.get("result_remote_id")
        psindex_data = json.loads(get_content(remote_id))
        status = resjson.get("status")

        if status == 'failed':
            msg = 'error_message'
            return error_result_json("100003", '当前任务失败:[{}]'.format(msg))
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100002", f'输入引脚input获取失败:[{str(ex)}]')

    try:
        result_data = psindex_data
        logger.debug("======>>>>>>> result_data:{}".format(result_data))
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100004", '查询特征变量分箱失败:[{}]'.format(str(ex)))
    return succ_result_json(result_data)


@hetero_psindex_app.route("/analysis/psi", methods=["POST"])
def psi_ok():
    """ PSI修改中弹出界面中的 确认按钮"""
    try:
        logger.info("==>> PSI修改中弹出界面中的 确认按钮 ...")
        req_data = request.data
        if req_data is None:
            return error_result_json('100000', ERROR_MISSING_PARAMS)
        params = json.loads(req_data)
        logger.debug("psi_ok params:{}".format(params))
        flow_id = params.get('flowId')
        cid = params.get('containerId')
        check_id(flow_id, cid)
        bin_id = params.get('binId')
        feature_name = params.get('featureName')
        feature_type = params.get('featureType')
        nid = params.get('nid')
        bin_param = params.get('binParam')

        logger.debug('feature_name：{}, nid：{}, binId：{}, binParam:{}'.format(
            feature_name, nid, bin_id, bin_param))
    except Exception as ex:
        return error_result_json("100001", f"参数获取异常[flowId/containerId].{str(ex)}")

    try:
        ware_id = params.get('wareId')
        logger.debug(f'flowId：{flow_id}, containerId：{cid}, wareId：{ware_id}')
        ctx = WareContext(flow_id, ware_id, cid)
        psindex_ware_job = ctx.get_ware_data('psindex_ware_job')
        if isinstance(psindex_ware_job, str) and psindex_ware_job is not None:
            psindex_job = json.loads(psindex_ware_job)
            params = {'psi_params': {
                'binId': bin_id,
                'featureName': feature_name,
                'featureType': feature_type,
                'nid': nid,
                'binParam': bin_param
            }}
            logger.debug('psi_params params:{}'.format(params))
            psindex_job['settings'] = params
            logger.info('++++++++ware_job2++++++++: {}'.format(psindex_job))
            job_id = psindex_job['jobId']
            flow_id = psindex_job['flowId']
            container_id = psindex_job['containerId']
            ware_id = psindex_job['wareId']
            ware_version = psindex_job['wareVersion']
        else:
            return error_result_json("100002", "woe_ware_job不能为空, 请排查当前组件ware_data中是否存在.")
    except Exception as ex:
        return error_result_json("100003", str(ex))
    logger.info('outside callback')
    # 组件进入交互状态  目前前端存在的一个bug，故需要
    flow_callback(flow_id, container_id, ware_id, ware_version, job_id, 'PAUSE')
    # 组件进入运行状态
    flow_callback(flow_id, container_id, ware_id, ware_version, job_id, 'START')
    logger.debug('======>>>>>> before start job, psindex_job:{}'.format(psindex_job))
    psindex_train_ware_start(psindex_job)
    return succ_result_json(None)


@hetero_psindex_app.route("/analysis/psi/finishCompute", methods=["POST"])
def finish_compute():
    """ 完成PSI对应的按钮 """
    try:
        logger.info("==>> 完成psi计算...")
        req_data = request.data
        if req_data is None:
            return error_result_json('100000', ERROR_MISSING_PARAMS)
        params = json.loads(req_data)
        logger.debug("finish_compute params:{}".format(params))
        flow_id = params.get('flowId')
        cid = params.get('containerId')
        check_id(flow_id, cid)
    except Exception as ex:
        return error_result_json("100001", f"参数获取异常[flowId/containerId].{str(ex)}")
    try:
        ware_id = params.get('wareId')
        logger.debug(f'flowId：{flow_id}, containerId：{cid}, wareId：{ware_id}')
        ctx = WareContext(flow_id, ware_id, cid)
        # 从psindex_ware_job获取job信息
        psindex_ware_job = ctx.get_ware_data('psindex_ware_job')
        job_id = None
        ware_id = None
        if isinstance(psindex_ware_job, str) and psindex_ware_job is not None:
            psindex_job = json.loads(psindex_ware_job)
            job_id = psindex_job.get("jobId")
            ware_id = psindex_job.get("wareId")
            ware_version = psindex_job.get('wareVersion')
        logger.debug('jobId：{}, wareId：{}, wareVersion：{}'.format(job_id, ware_id, ware_version))
        if job_id is not None and ware_id is not None:
            # flow_callback(flow_id, cid, ware_id, ware_version, job_id, "PAUSE")
            flow_callback(flow_id, cid, ware_id, ware_version, job_id, "SUCCESS")
            return succ_result_json(None)
        else:
            flow_callback(flow_id, cid, ware_id, ware_version, job_id, "FAILED")
            return error_result_json("100002", "平台未查到参数[jobId/wareId/wareVersion]")
    except Exception as ex:
        return error_result_json("100003", str(ex))


@hetero_psindex_app.route("/analysis/psi/discrete", methods=["POST"])
def query_discrete_enums():
    """ 请求某个特征的离散值"""
    try:
        logger.info("==>> 查询离散型特征变量枚举值:[/analysis/psi/discrete]...")
        req_data = request.data
        if req_data is None:
            return error_result_json('100000', ERROR_MISSING_PARAMS)
        params = json.loads(req_data)
        logger.debug("query_discrete_enums params:{}".format(params))

        flow_id = params.get("flowId")
        cid = params.get("containerId")
        feature_name = params.get('featureName')
        nid = params.get('nid')
        check_id(flow_id, cid)
        logger.debug('nid：{}, featureName：{}'.format(nid, feature_name))
    except Exception as ex:
        return error_result_json("100001", f"参数获取异常[flowId/containerId].{str(ex)}")
    try:
        ware_id = WARE_ID
        logger.debug(f'flowId：{flow_id}, containerId：{cid}, wareId：{ware_id}')
        ctx = WareContext(flow_id, ware_id, cid)
        categorical_feature_enums = json.loads(ctx.get_ware_data("categorical_feature_enums"))
        logger.debug('获取发起方数据categorical_feature_enums:{}'.format(categorical_feature_enums))
    except Exception as ex:
        return error_result_json("100002", f'输入引脚input获取失败:[{str(ex)}]')

    try:
        categorical_feature_currnid = categorical_feature_enums.get(nid)
        if categorical_feature_currnid is None:
            raise ValueError("离散特征汇总中没有{}节点的离散特征".format(nid))
        feature_info = categorical_feature_currnid.get(feature_name)
        if feature_info is None:
            raise ValueError("{}节点中没有离散特征{}".format(nid, feature_name))
        enums_is_big = feature_info.get('num_enums_big')
        if enums_is_big:
            logger.info("该离散型特征变量枚举值超过700, 暂不支持手动分箱")
            return error_result_json("100004", '该离散型特征变量枚举值超过700, 暂不支持手动分箱.')
        else:
            data_enums = feature_info.get('enums')
            logger.debug("data_enums:{}".format(data_enums))
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100003", '查询离散型特征变量枚举值失败:[{}]'.format(str(ex)))
    return succ_result_json(data_enums)


def psindex_train_ware_start(job):
    server_url = RouterService.get_core()
    if server_url is not None:
        url = 'http://' + server_url + '/ware/algorithm/start'
        logger.info("flow callback to:{}, data:{}".format(url, job))
        res = HttpUtil.post(url, json.dumps(job))
        logger.info("flow callback res:{}".format(res))
        return res
    logger.info("===>> 重新启动ware失败，url为None")


def check_id(flow_id, container_id):
    if flow_id is None:
        return error_result_json('100000', 'missing param flowId')
    if container_id is None:
        return error_result_json('100000', 'missing param containerId')


class HeteroPSIndexWare(BaseWare):
    """
    MissingWare
    """

    def __init__(self, setting_file: str):
        # 设置角色字典，注意是类型不是对象
        roles = {
            Role.GUEST: HeteroPSIGuest,
            Role.HOST: HeteroPSIHost,
            Role.NODATA: HeteroPSINodata
        }
        super().__init__(setting_file, roles)

    def build_algo_data_transfer(self, job: Job, role_parser: JobRoleParser) -> PSIndexAlgoDataTransfer:
        return PSIndexAlgoDataTransfer(self.all_roles(), job, role_parser)

    def build_role_parser(self, job: Job) -> JobRoleParser:
        return HeteroModelJobRoleParser(job)

    def after_start(self, job: Job, ctx: Context):
        # 组件控制状态
        pass
