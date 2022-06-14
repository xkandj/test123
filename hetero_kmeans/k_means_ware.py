# coding: utf-8
from flask import Blueprint

from fmpc.base.Node import Job
from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_kmeans.k_means_data_transfer import KMeansDataTransfer
from wares.hetero_kmeans.k_means_guest import KMeansGuest
from wares.common.algo_data_transfer import BaseAlgorithmDataTransfer
from wares.common.base_ware import BaseWare
from wares.common.job_role_parser import JobRoleParser
from wares.hetero_kmeans.k_means_host import KMeansHost
from fmpc.web.ware_app_base import WareAppBase
from wares.common.web.machine_learning_ware_app import MachineLearningWareApp

kmeans_app = Blueprint('kmeans_ware_app', __name__, static_folder='h', static_url_path='/h')
logger = get_fmpc_logger(__name__)
WARE_ERROR_CODE = -1


class KMeansJobRoleParser(JobRoleParser):

    def parse_role_dict(self, job):
        nid_role_dict = {}
        role_nodes_dict = {}
        owner = None
        if len(job.flnodes) > 1:
            raise ValueError("k-means不支持三方或无数据方学习")

        for node in (job.flnodes + [job.currnode]):
            if node.is_owner:
                owner = node
            role = self.get_role_from_node(node, job)
            nid_role_dict[(node.node.nid, node.node.serial_id)] = role
            if role in role_nodes_dict:
                role_nodes_dict[role].append(node)
            else:
                role_nodes_dict[role] = [node]
        return nid_role_dict, role_nodes_dict, owner

    def get_role_from_node(self, node, job):
        if node.is_owner:
            return Role.GUEST
        else:
            return Role.HOST


class KMeansWare(BaseWare):
    """
    KMeansWare
    """

    def __init__(self, setting_file):
        roles = {
            "guest": KMeansGuest,
            "host": KMeansHost,
        }
        super().__init__(setting_file, roles)

    def build_role_parser(self, job: Job) -> JobRoleParser:
        return KMeansJobRoleParser(job)

    def build_algo_data_transfer(self, job: Job, role_parser: JobRoleParser) -> BaseAlgorithmDataTransfer:
        return KMeansDataTransfer(self.all_roles(), job, role_parser)

    def build_ware_app(self) -> WareAppBase:
        return MachineLearningWareApp(self)