# coding: utf-8
from flask import Blueprint

from fmpc.base.Node import Job
from fmpc.fl.consts import Role
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.common.base_ware import BaseWare
from wares.common.job_role_parser import JobRoleParser
from wares.hetero_kmeans.k_means_data_transfer import KMeansDataTransfer
from wares.hetero_kmeans.k_means_predict_guest import KMeansGuestPred
from wares.hetero_kmeans.k_means_predict_host import KMeansHostPred
from wares.hetero_kmeans.k_means_ware import KMeansJobRoleParser

k_means_predict_app = Blueprint('k_means_predict_ware_app', __name__, static_folder='h', static_url_path='/h')
logger = get_fmpc_logger(__name__)

class KMeansPredictWare(BaseWare):
    def __init__(self, setting_file: str):
        # 设置角色字典，注意是类型不是对象
        roles = {
            Role.GUEST: KMeansGuestPred,
            Role.HOST: KMeansHostPred,
        }
        super().__init__(setting_file, roles)

    def build_algo_data_transfer(self, job: Job, role_parser: JobRoleParser) -> KMeansDataTransfer:
        return KMeansDataTransfer(self.all_roles(), job, role_parser)

    def build_role_parser(self, job: Job) -> JobRoleParser:
        return KMeansJobRoleParser(job)
