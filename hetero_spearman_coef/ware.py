# -*- coding: utf-8 -*-
import json
import traceback
from flask import Blueprint, request, make_response
from fmpc.base.Node import Job, Node
from fmpc.base.ResultModels import error_result_json, result_json
from fmpc.base.WareContext import WareContext
from fmpc.utils.FastDfsSliceUtils import get_bytes
from fmpc.utils.HttpUtil import HttpUtil
from fmpc.utils.LogUtils import get_fmpc_logger
from fmpc.utils.RouterService import RouterService
from wares.common.base_ware import BaseWare
from wares.common.job_role_parser import JobRoleParser
from wares.hetero_spearman_coef.const import DOWNLOAD_RESULT, OWNER, PARTICIPANT
from wares.hetero_spearman_coef.spearman_algo_data_transfer import HeteroSpearmanAlgoDataTransfer
from wares.hetero_spearman_coef.spearman_onwer import SpearmanOwner
from wares.hetero_spearman_coef.spearman_parti import SpearmanParticipant

ware_app = Blueprint('hetero_spearman_coef_ware_app',
                     __name__, static_folder='h', static_url_path='/h')
logger = get_fmpc_logger(__name__)


def check_id(container_id_, flow_id_, ware_id_):
    logger.debug("接口子函数check_id")
    if container_id_ is None:
        return '10000', 'missing param containerId'
    if flow_id_ is None:
        return '10000', 'missing param flowId'
    if ware_id_ is None:
        return '10000', 'missing param wareId'
    return '0', 'ok'


def deduce_post_param(request):
    logger.debug("接口子函数deduce_post_param")
    data_ = request.data
    logger.info("request data:{}".format(data_))
    if data_ is None:
        return error_result_json('10000', 'missing params')
    params_ = json.loads(data_)
    ware_id_ = params_.get('wareId')
    flow_id_ = params_.get('flowId')
    version_ = params_.get('version')
    container_id_ = params_.get('containerId')
    code_tag, msg_ = check_id(container_id_, flow_id_, ware_id_)
    if code_tag != '0':
        return error_result_json(code_tag, msg_)
    return ware_id_, flow_id_, version_, container_id_, params_


def get_all_columns(ware_input_):
    """ 得到特征信息列表"""
    logger.debug("接口子函数get_all_columns")
    nodes_lists_ = ware_input_.get('nodes')
    res_feature_columns = []
    for node_input_ in nodes_lists_:
        node_info_ = node_input_.get('node')
        node_name_ = node_info_.get('nodeName')
        nid_ = node_info_.get('nid')
        s_id_ = node_info_.get('serialId')
        node_type_ = node_input_.get('nodeType')
        dataset_ = node_input_.get('dataset')
        if dataset_ is None:
            continue
        if node_type_ == 'OWNER' or node_type_ == 'PARTICIPATION':
            dataset_meta_columns_ = dataset_.get('meta').get('columns')
            logger.info('===>>节点nid:{},sid:{},columns:{}'.format(
                nid_, s_id_, json.dumps(dataset_meta_columns_)))
            for fea in dataset_meta_columns_:
                if fea.get('columnType') is None:  # 该字段为none，说明是特征列
                    cur_feature = {
                        'name': fea.get('name'),
                        'iv': fea.get('iv'),
                        'importance': fea.get('importance'),
                        'type': fea.get('type'),
                        'distribution': fea.get('distribution'),
                        'nid': nid_,
                        'nodeName': node_name_,
                        "selected": False
                    }
                    res_feature_columns.append(cur_feature)
    return res_feature_columns


def update_cols(all_cols, selected_cols):
    logger.debug("接口子函数update_cols")
    for s_col in selected_cols:
        for a_col in all_cols:
            if (s_col.get('name') == a_col.get('name')) and (s_col.get('nid') == a_col.get('nid')):
                a_col['selected'] = s_col['selected']


def ware_start(job_dict):
    logger.debug("接口子函数ware_start")
    server_url = RouterService.get_core()
    if server_url is not None:
        url = 'http://' + server_url + '/ware/algorithm/start'
        logger.info("重新启动ware start, url:{}, request:{}\n\n".format(url, json.dumps(job_dict)))
        res = HttpUtil.post(url, json.dumps(job_dict))
        logger.info("重新启动ware start, response:{}\n\n".format(json.dumps(res)))
        return res
    logger.info('重新启动ware start失败, url为None')
    return None


# 1，获取spearman组件的特征列表-接口
@ware_app.route('/spearman/getDatasetColumns', methods=['POST'])
def get_dataset_columns():

    logger.debug("接口函数get_dataset_columns")
    try:
        ware_id_, flow_id_, version_, container_id_, _ = deduce_post_param(request)
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100001", "参数获取异常[flowId/containerId].{}".format(str(ex)))

    try:
        ctx_ = WareContext(flow_id_, ware_id_, container_id_)
        ware_input_ = ctx_.get_ware_input_data("input1")
    except Exception as ex:
        logger.error(format(traceback.format_exc()))
        return error_result_json("100002", '获取输入引脚input失败:[{}]'.format(str(ex)))

    try:
        all_columns_ = get_all_columns(ware_input_)

        # 判断spearman计算状态
        fea_columns_ = ctx_.get_ware_data("featureColumns")
        if fea_columns_:
            update_cols(all_columns_, fea_columns_)

    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100003", 'spearman查询特征变量失败:[{}]'.format(str(ex)))

    return result_json("0", "OK", all_columns_)


# 2，计算spearman-接口
@ware_app.route('/spearman/compute', methods=['POST'])
def compute():
    logger.debug("接口函数compute")
    logger.info('开始计算spearman相关系数...')
    try:        
        ware_id_, flow_id_, version_, container_id_, params_ = deduce_post_param(request)
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100001", "参数获取异常[flowId/containerId].{}".format(str(ex)))

    try:
        feature_columns_ = params_.get('featureColumns')
        if not feature_columns_:
            logger.info("没有选择特征，请选择特征!")
            return error_result_json("100001", "参数获取异常[featureColumns]没选特征")

        ctx_ = WareContext(flow_id_, ware_id_, container_id_)
        ctx_.set_ware_data("featureColumns", json.dumps(feature_columns_))
        job_ = ctx_.get_ware_data('spearman_ware_job')
        if isinstance(job_, str) and job_ is not None:
            restart_job_dict = json.loads(job_)
            restart_job_dict['settings'] = {"featureColumns":feature_columns_}
            restart_job_dict['wareServerType'] = 'training'
        else:
            return error_result_json("100002", "spearman_ware_job不能为空, 请排查当前组件ware_data中是否存在.")

        # 重启warestart
        ware_start(restart_job_dict)
        return result_json("0", 'OK', {'jobId': job_.job_id})
    except Exception as ex:
        logger.info(format(traceback.format_exc()))
        return error_result_json("100003", str(ex))


# 3，下载结果-接口
@ware_app.route('/spearman/download', methods=['POST'])
def download():
    logger.debug("接口函数download")
    ware_id, flow_id, version, container_id, _ = deduce_post_param(request)
    ctx = WareContext(flow_id, ware_id, container_id)
    rs = ctx.get_ware_data(DOWNLOAD_RESULT)
    rs = json.loads(rs)
    remote_id = rs["spearman_matrix_remote_id"]
    logger.info("======>>> remote_id:{}".format(remote_id))
    content = get_bytes(remote_id)
    response = make_response(content)
    file_name = "application/octet-stream"
    response.headers['Content-Type'] = file_name
    response.headers['Content-Disposition'] = "inline; filename=" + file_name
    return response

class HeteroSpearmanRoleParser(JobRoleParser):
    def _set_role_nodes_dict(self, role: str, flnode: Node, role_nodes_dict: dict) -> dict:
        """
        设置role_nodes_dict. 该dict key为节点角色, value为Node列表

        :param role: 角色
        :param flnode: 节点
        :param role_nodes_dict: 角色对应的节点
        :return: role_nodes_dict
        """
        logger.debug("_set_role_nodes_dict, role:{}, flnode:{}, role_nodes_dict{}".format(role, flnode, role_nodes_dict))
        if role_nodes_dict.get(role) is None:
            role_nodes_dict[role] = [flnode]
        else:
            role_nodes_dict[role].append(flnode)
        return role_nodes_dict

    def parse_role_dict(self, job: Job):
        """
        根据job解析role

        :param job: 任务
        :return: nid_role_dict: 节点id对应的角色, key为(nid, serial_id), value为角色
        :return: role_nodes_dict: 角色对应的节点, key为角色, value为Node列表
        """
        logger.debug("parse_role_dict")
        all_nodes = job.flnodes + [job.currnode]
        all_nodes.sort(key=lambda x: x.node.nid)  # sort by nid
        nid_role_dict = {}
        role_nodes_dict = {}
        owner = None
        for flnode in all_nodes:
            k = (flnode.node.nid, flnode.node.serial_id)
            if flnode.is_owner:
                owner = flnode
                nid_role_dict[k] = OWNER
                self._set_role_nodes_dict(OWNER, flnode, role_nodes_dict)
            else:
                nid_role_dict[k] = PARTICIPANT
                self._set_role_nodes_dict(PARTICIPANT, flnode, role_nodes_dict)
        return nid_role_dict, role_nodes_dict, owner


class HeteroSpearmanCoefWare(BaseWare):
    """
    创建HeteroSpearmanCoefWare对象
    """

    def __init__(self, setting_file):
        roles = {
            OWNER: SpearmanOwner,
            PARTICIPANT: SpearmanParticipant
        }
        super().__init__(setting_file, roles)

    def build_role_parser(self, job: Job) -> HeteroSpearmanRoleParser:
        return HeteroSpearmanRoleParser(job)

    def build_algo_data_transfer(self, job: Job, role_parser: JobRoleParser) -> HeteroSpearmanAlgoDataTransfer:
        return HeteroSpearmanAlgoDataTransfer(self.all_roles(), job, role_parser)
