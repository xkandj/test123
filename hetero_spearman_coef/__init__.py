# coding: utf-8
import os

from fmpc.web.AvatarApplication import registry
from wares.hetero_spearman_coef.ware import HeteroSpearmanCoefWare, ware_app
from fmpc.utils.LogUtils import get_fmpc_logger

__logger = get_fmpc_logger(__name__)
__logger.info('HeteroSpearmanCoefWare init')
setting_file = os.path.join(os.path.dirname(__file__), 'ware.json')
registry(ware_app, HeteroSpearmanCoefWare(setting_file))
__logger.info('HeteroSpearmanCoefWare init end')

