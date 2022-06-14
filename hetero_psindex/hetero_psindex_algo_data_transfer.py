import typing

from fmpc.utils import ConfigUtils
from fmpc.base.Node import Job
from wares.common.job_role_parser import JobRoleParser
from wares.common.local_develop_helper import LocalDevAlgorithmDataTransfer


class PSIndexAlgoDataTransfer(LocalDevAlgorithmDataTransfer):

    def __init__(self, all_roles: typing.List[str], job: Job, role_parser: JobRoleParser) -> None:
        super().__init__(all_roles, job, role_parser)
        # interactive layer
        timeout_factor = ConfigUtils.get_config('python', 'timeoutfactor')
        if timeout_factor is not None and len(timeout_factor) > 0:
            timeout_factor = int(timeout_factor)
        else:
            timeout_factor = 3
        # 组件初始启动时，host方将本方的分箱信息发送给host（经过iv组件后的分箱信息，是否包含特征及分箱信息）
        self.target_feature_info_to_guest = self.create_data_event('target_feature_info_to_guest_{}', src=['host'],
                                                                   dst=['guest'], timeout=3600 * timeout_factor)

        # 组件初始启动时，guest方将确定的待处理特征及相关信息发送给host方
        self.target_feature_info = self.create_data_event('target_info_key', src=['guest'], dst=['host'],
                                                          timeout=3600 * timeout_factor)
        # # 组件初始启动时，guest方无特征，则需要host将特征相关信息发送给其他方
        # self.target_feature_info_to_others = self.create_data_event('target_info_others', src=['host'],
        #                                                             dst=['guest', 'host'],
        #                                                             timeout=3600 * timeout_factor)
        # host方将本方含有iv分箱信息的特征发送给guest方
        self.iv_bin_in_currnid_to_guest = self.create_data_event('iv_bin_in_currnid_to_guest_{}', src=['host'],
                                                                   dst=['guest'], timeout=3600 * timeout_factor)
        # guest方汇总所有方含有分箱信息的特征，发送给host方
        self.iv_bin_features = self.create_data_event('iv_bin_features', src=['guest'], dst=['host'],
                                                          timeout=3600 * timeout_factor)

        self.encrypted_y = self.create_data_event('data_y_encrypted', src=['guest'], dst=['host'],
                                                  timeout=3600 * timeout_factor)
        # 卡方分箱值
        self.chi_val_dict = self.create_data_event('chi_val_dict_count_{}', src=['host'],
                                                   dst=['guest'], timeout=3600 * timeout_factor)
        # 最小卡方值
        self.min_chi_val = self.create_data_event('min_chi_val_count_{}', src=['guest'], dst=['host'],
                                                  timeout=3600 * timeout_factor)
        # 计算卡方是否继续运行标识符
        self.is_do = self.create_data_event('is_stop_cal_chi_val_count_{}', src=['host'],
                                            dst=['guest'], timeout=3600 * timeout_factor)
        # 当计算稳定性指标的特征在host方时，需要将
        self.psi_json = self.create_data_event('psi_json', src=['guest', 'host'], dst=['owner'],
                                               timeout=3600 * timeout_factor)

        # 所有节点的离散元素汇总发给发起方
        self.categorical_feature_enums = self.create_data_event('categorical_feature_enums_currnid_{}',
                                                                src=['guest', 'host'],
                                                                dst=['owner'], timeout=3600 * timeout_factor)

        # 分箱里面用的的事件
        self.features_event = self.create_data_event("{}_{}_features", src=['host'], dst=['guest'],
                                                     timeout=3600 * timeout_factor)

        self.bins_event = self.create_data_event("{}_{}_bins", src=['guest'], dst=['host'],
                                                 timeout=3600 * timeout_factor)

        self.features_a_event = self.create_data_event("{}_{}_features_a", src=['host'],
                                                       dst=['guest'], timeout=3600 * timeout_factor)

        self.group_dict_event = self.create_data_event("{}_{}_group_dict", src=['host'],
                                                       dst=['guest'], timeout=3600 * timeout_factor)

        self.finish_event = self.create_data_event("{}_{}_finish_{}", src=['guest'],
                                                   dst=['host'], timeout=3600 * timeout_factor)

        self.min_chi2_index_event = self.create_data_event("{}_{}_minChi2_index_{}", src=['guest'],
                                                           dst=['host'], timeout=3600 * timeout_factor)

        self.update_dict_event = self.create_data_event("{}_{}_update_dict_{}", src=['host'],
                                                        dst=['guest'], timeout=3600 * timeout_factor)

        self.key_delete_event = self.create_data_event("{}_{}_key_delete_{}", src=['host'],
                                                       dst=['guest'], timeout=3600 * timeout_factor)
