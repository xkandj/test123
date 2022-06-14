from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_spearman_coef.const import PARTICIPANT
from wares.hetero_spearman_coef.spearman_base import SpearmanBase

logger = get_fmpc_logger(__name__)

class SpearmanParticipant(SpearmanBase):
    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)

    def static_role(self) -> str:
        return PARTICIPANT
