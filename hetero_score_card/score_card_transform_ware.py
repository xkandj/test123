# coding: utf-8
import json
import typing

from flask import Blueprint, request

from fmpc.base.Node import Job, NodeJobRole
from fmpc.base.ResultModels import error_result_json, succ_result_json
from fmpc.base.WareContext import WareContext
from fmpc.fl.consts import Role, Data, InputType
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_ware import BaseWare
from wares.common.fmpc_error import InvalidNodeDataTypeError
from wares.common.job_role_parser import JobRoleParser, INVALID_DATA_TYPE_MSG
from wares.hetero_score_card.score_card_transform_algo_data_transfer import ScoreCardTransformAlgoDataTransfer
from wares.hetero_score_card.score_card_transform_base import SCORE_CARD_WARE_ID, SCORE_CARD_REPORT, MODEL_INPUT, \
    MODEL_OUTPUT
from wares.hetero_score_card.score_card_transform_guest import ScoreCardTransformGuest
from wares.hetero_score_card.score_card_transform_host import ScoreCardTransformHost
from wares.hetero_score_card.score_card_transform_nodata import ScoreCardTransformNodata

score_card_app = Blueprint('score_card_app', __name__, static_folder='h', static_url_path='/h')
logger = get_fmpc_logger(__name__)


WARE_ERROR_CODE = -1
CHECK_INPUTS_MSG = "评分卡组件输入不符合要求，请检查！"
INPUT_CATEGORY = "模型|监督算法|分类模型|二分类"


@score_card_app.route('/model_report', methods=['POST'])  # model_report
def model_report():
    """模型报告"""
    req_data = request.data
    if req_data is None:
        return error_result_json('10000', 'missing params')
    params = json.loads(req_data)
    flow_id = params.get('flowId')
    if flow_id is None:
        return error_result_json('10000', 'missing param flowId')
    cid = params.get('containerId')
    if cid is None:
        return error_result_json('10000', 'missing param containerId')

    ctx = WareContext(flow_id, SCORE_CARD_WARE_ID, cid)
    rs = ctx.get_ware_output_data(SCORE_CARD_REPORT)
    if rs is None:
        return error_result_json('10000', '报告还未生成')
    return succ_result_json(rs)


class ScoreCardTransformJobRoleParser(JobRoleParser):
    """
    一般的纵向的机器学习组件roleParser
    """

    def parse_role_dict(self, job: Job) -> typing.Tuple[
        typing.Dict[typing.Tuple[str, str], str], typing.Dict[str, typing.List[NodeJobRole]], NodeJobRole]:
        """
        纵向预测时使用的parser
        @param job:
        @return:
        """

        nid_role_dict = {}
        role_nodes_dict = {}
        owner = None
        model_input = self._get_model_input(job)
        for node in (job.flnodes + [job.currnode]):
            if node.is_owner:
                # 节点是ow
                owner = node
            # 判断角色的columns是不是只有id列，没有特征列
            features, data_type = self._get_node_features(model_input, node)
            if len(features) == 0 and node.is_owner:
                # 发起方无数据的情况，视作nodata
                role_nodes_dict[Role.NODATA] = [node]
                nid_role_dict[(node.node.nid, node.node.serial_id)] = Role.NODATA
                continue
            node.data_type = data_type
            role = self.get_role_from_node(node)
            nid_role_dict[(node.node.nid, node.node.serial_id)] = role
            if role in role_nodes_dict:
                role_nodes_dict[role].append(node)
            else:
                role_nodes_dict[role] = [node]
        return nid_role_dict, role_nodes_dict, owner

    def get_role_from_node(self, node: NodeJobRole) -> str:
        """
        仅根据node的数据集类型判断
        :param node:
        :return:
        """
        data_type = node.data_type
        if data_type == Data.NONEDATA or data_type == Data.NODATA:
            return Role.NODATA
        if data_type == Data.XY or data_type == Data.Y:
            return Role.GUEST
        elif data_type == Data.X:
            return Role.HOST
        raise InvalidNodeDataTypeError(INVALID_DATA_TYPE_MSG % data_type)

    def _get_model_input(self, job: Job):
        model_input_list = [i for i in job.inputs if i.param_type == InputType.MODEL]
        if len(model_input_list) == 0:
            raise InvalidNodeDataTypeError("未找到数据集类型输入！")
        else:
            return model_input_list[0]

    def _get_node_features(self, model_input, node: NodeJobRole):
        input_nodes = model_input.params['nodes']
        ret = []
        node_type = None
        for n in input_nodes:
            if n['node']['nid'] != node.node.nid:
                continue
            if 'model' in n:
                model = n['model']
                if 'meta' in model:
                    meta = model['meta']
                    if 'datasetType' in meta:
                        node_type = meta['datasetType']
                        if 'columns' in meta:
                            columns = meta['columns']
                            for c in columns:
                                if 'columnType' in c:
                                    type_ = c['columnType']
                                    if type_ is not None:
                                        if type_ == 'matchColumn':
                                            continue
                                        else:
                                            ret.append(c)
                                    else:
                                        ret.append(c)
                                else:
                                    ret.append(c)
        return ret, node_type


class ScoreCardTransformWare(BaseWare):

    def __init__(self, setting_file: str):
        # 设置角色字典，注意是类型不是对象
        roles = {
            Role.GUEST: ScoreCardTransformGuest,
            Role.HOST: ScoreCardTransformHost,
            Role.NODATA: ScoreCardTransformNodata
        }
        super().__init__(setting_file, roles)

    def build_algo_data_transfer(self, job: Job, role_parser: JobRoleParser) -> ScoreCardTransformAlgoDataTransfer:
        return ScoreCardTransformAlgoDataTransfer(self.all_roles(), job, role_parser)

    def build_role_parser(self, job: Job) -> JobRoleParser:
        return ScoreCardTransformJobRoleParser(job)


