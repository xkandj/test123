import typing
from fmpc.base.Node import Job
from fmpc.utils import ConfigUtils
from wares.common.job_role_parser import JobRoleParser
from wares.common.local_develop_helper import LocalDevAlgorithmDataTransfer
from wares.hetero_spearman_coef.const import DEFAULT_TIMEOUT_FACTOR


class HeteroSpearmanAlgoDataTransfer(LocalDevAlgorithmDataTransfer):

    def __init__(self, all_roles: typing.List[str], job: Job, role_parser: JobRoleParser) -> None:
        super().__init__(all_roles, job, role_parser)
        # interactive layer
        timeout_factor = ConfigUtils.get_config('python', 'timeoutfactor')
        if timeout_factor is not None and len(timeout_factor) > 0:
            timeout_factor = int(timeout_factor)
        else:
            timeout_factor = DEFAULT_TIMEOUT_FACTOR
        # {from_nid}_{to_nid}_rank_e
        self.rank_e = self.create_data_event('{}_{}_rank_e', src=['owner', 'participant'], dst=['participant'], timeout=3600 * timeout_factor)
        # {from_nid}_{to_nid}_inter_matrix_e
        self.inter_matrix_e = self.create_data_event('{}_{}_inter_matrix_e', src=['participant'], dst=['owner', 'participant'], timeout=3600 * timeout_factor)
        # {from_nid}_assemble_matrix
        self.assemble_matrix = self.create_data_event('{}_assemble_matrix', src=['participant'], dst=['owner'], timeout=3600 * timeout_factor)




