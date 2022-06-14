import typing

from fmpc.utils import ConfigUtils

from fmpc.base.Node import Job
from wares.common.job_role_parser import JobRoleParser
from wares.common.local_develop_helper import LocalDevAlgorithmDataTransfer


DEFAULT_TIMEOUT_FACTOR = 3

class KMeansDataTransfer(LocalDevAlgorithmDataTransfer):

    def __init__(self, all_roles: typing.List[str], job: Job, role_parser: JobRoleParser) -> None:
        super().__init__(all_roles, job, role_parser)
        timeout_factor = ConfigUtils.get_config('python', 'timeoutfactor')
        if timeout_factor is not None and len(timeout_factor) > 0:
            timeout_factor = int(timeout_factor)
        else:
            timeout_factor = DEFAULT_TIMEOUT_FACTOR
        # {from_nid}_{n_init}_{k_cluster}_xxxx
        self.kmeans_plusplus_init_distance = self.create_data_event('{}_{}_{}_kmeans_plusplus_init_distance',
                                                                    src=['host'], dst=['guest'],
                                                                    timeout=3600 * timeout_factor)
        # {from_nid}_{n_init}_{k_cluster}_xxxx
        self.kmeans_plusplus_init_idx = self.create_data_event('{}_{}_{}_kmeans_plusplus_init_idx', src=['guest'],
                                                               dst=['host'],
                                                               timeout=3600 * timeout_factor)
        # {from_nid}_{n_init}_{iter_i}_xxxx
        self.kmeans_single_distance = self.create_data_event('{}_{}_{}_kmeans_single_distance', src=['host'],
                                                             dst=['guest'],
                                                             timeout=3600 * timeout_factor)
        # {from_nid}_{n_init}_{iter_i}_xxxx
        self.kmeans_single_relocate_labels = self.create_data_event(
            '{}_{}_{}_kmeans_single_relocate_labels', src=['guest'], dst=['host'], timeout=3600 * timeout_factor)

        # {from_nid}_{n_init}_{iter_i}_kmeans_single_center_shift_b
        self.kmeans_single_center_shift_b = self.create_data_event(
            '{}_{}_{}_kmeans_single_center_shift_b', src=['host'], dst=['guest'], timeout=3600 * timeout_factor)

        # {from_nid}_{n_init}_{iter_i}_xxxx
        self.kmeans_single_center_shift_all = self.create_data_event(
            '{}_{}_{}_kmeans_single_center_shift_all', src=['guest'], dst=['host'], timeout=3600 * timeout_factor)

        # {from_nid}_xxx
        self.kmeans_evaluate_intra_and_inter_dist_b = self.create_data_event(
            '{}_kmeans_evaluate_intra_and_inter_dist_b', src=['host'], dst=['guest'], timeout=3600 * timeout_factor)

        # {from_nid}_xxx
        self.kmeans_single_predict_distance = self.create_data_event(
            '{}_kmeans_single_predict_distance', src=['host'], dst=['guest'], timeout=3600 * timeout_factor)

        # 
        self.kmeans_model_report_body = self.create_data_event(
            'kmeans_model_eval_report', src=['guest'], dst=['owner'], timeout=3600 * timeout_factor)