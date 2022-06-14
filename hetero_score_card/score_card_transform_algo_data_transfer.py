import typing

from fmpc.utils import ConfigUtils
from fmpc.base.Node import Job
from wares.common.job_role_parser import JobRoleParser
from wares.common.local_develop_helper import LocalDevAlgorithmDataTransfer
from wares.hetero_score_card.const import DEFAULT_TIMEOUT_FACTOR


class ScoreCardTransformAlgoDataTransfer(LocalDevAlgorithmDataTransfer):

    def __init__(self, all_roles: typing.List[str], job: Job, role_parser: JobRoleParser) -> None:
        super().__init__(all_roles, job, role_parser)
        timeout_factor = ConfigUtils.get_config('python', 'timeoutfactor')
        if timeout_factor is not None and len(timeout_factor) > 0:
            timeout_factor = int(timeout_factor)
        else:
            timeout_factor = DEFAULT_TIMEOUT_FACTOR

        self.min_max_score = self.create_data_event('min_max_score', src=['guest'], dst=['others'],
                                                           timeout=3600)
        self.xbup = self.create_data_event('Xbup_{}_{}_{}_{}', src=['host'], dst=['guest'],
                                           timeout=180 * 20)
        self.probscore = self.create_data_event('Probscore_{}_{}_{}_{}', src=['host'], dst=['guest'],
                                           timeout=180 * 20)

        self.owner_info_json = self.create_data_event('owner_info_json', src=['guest'], dst=['owner'],
                                                      timeout=600 * timeout_factor)
        self.owner_info_json_evaluate = self.create_data_event('owner_info_json_evaluate', src=['guest'], dst=['owner'],
                                                               timeout=3000 * timeout_factor)

        self.owner_info_json_predict = self.create_data_event('owner_info_json_predict', src=['guest'], dst=['owner'],
                                                              timeout=3000 * timeout_factor)