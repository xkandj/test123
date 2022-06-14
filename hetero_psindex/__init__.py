# coding: utf-8
import os
from fmpc.web.AvatarApplication import registry
from fmpc.utils.LogUtils import get_fmpc_logger

from wares.hetero_psindex.hetero_psindex_ware import HeteroPSIndexWare, hetero_psindex_app

# Population  Stability  Index
__logger = get_fmpc_logger(__name__)
__logger.info("HeteroPopulationStabilityIndexWare init")
setting_file = os.path.join(os.path.dirname(__file__), "ware.json")
registry(hetero_psindex_app, HeteroPSIndexWare(setting_file))
__logger.info("HeteroPopulationStabilityIndexWare init end")


