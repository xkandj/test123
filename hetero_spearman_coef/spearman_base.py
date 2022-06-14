from abc import ABC
import json
import numpy as np
import pandas as pd
from phe import paillier
from wares.common.base_algorithm import BaseAlgorithm
from fmpc.utils.JsonUtil import json_serialize_job
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_spearman_coef.const import DOWNLOAD_RESULT, OUTPUT

logger = get_fmpc_logger(__name__)


class SpearmanBase(BaseAlgorithm, ABC):

    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)
        self.selected_cols = []
        self.prev_nids = []
        self.next_nids = []
        self.all_nids = []
        self.local_matrix = None
        self.inter_matrix_list = []

    def do_ready(self) -> None:

        logger.debug("do_ready")
        self.parse_config()
        self.parse_settings()
        if not self.first_run:
            self.parse_dataset()

    def parse_config(self):
        self.nid = self.job.currnode.node.nid
        all_nodes = [self.job.currnode] + self.job.flnodes
        for nn in all_nodes:
            if nn.is_owner:
                self.owner_nid = nn.node.nid

    def parse_settings(self):

        logger.debug("parse_settings")
        settings = self.job.settings
        logger.debug("settings: {}".format(settings))
        if (settings is None) or (settings.get("featureColumns") is None):
            self.first_run = True
            self.ware_ctx.set_ware_data("spearman_ware_job", json_serialize_job(self.job))
        else:
            self.first_run = False
            logger.debug("featureColumns: {}".format(settings.get('featureColumns')))
            self.parse_feature_columns(settings.get('featureColumns'))
        logger.debug("self.first_run: {}".format(self.first_run))

    def parse_feature_columns(self, feature_columns):
        '''
        解析从settings中获取的特征列信息
        得到self.selected_cols, self.prev_nids, self.next_nids, self.all_nids
        :param feature_columns: list of dict
        '''
        logger.debug("parse_feature_columns")
        if len(feature_columns) == 0:
            logger.error("featureColumns is empty")
            raise ValueError("featureColumns is empty")

        # 被选中的节点的nid列表
        logger.debug("self.curr_nid: {}".format(self.curr_nid))
        nids = []
        for feature in feature_columns:
            selected = feature.get('selected')
            nid = feature.get('nid')
            name = feature.get('name')
            if selected and (nid == self.curr_nid):
                self.selected_cols.append(name)
            if nid not in nids:
                nids.append(nid)
        logger.debug("self.selected_cols: {}".format(self.selected_cols))

        # 如果owner被选中，则排在第一位
        owner_nid = self.owner_nid
        if owner_nid in nids:
            nids.pop(nids.index(owner_nid))
            nids.insert(0, owner_nid)
        self.all_nids = nids
        logger.debug("self.all_nids: {}".format(self.all_nids))

        # 前面节点nid列表，后面节点nid列表
        if self.curr_nid in self.all_nids:
            idx = self.all_nids.index(self.curr_nid)
            self.prev_nids = self.all_nids[:idx]
            self.next_nids = self.all_nids[idx+1:]
        
        logger.debug("self.prev_nids: {}".format(self.prev_nids))
        logger.debug("self.next_nids: {}".format(self.next_nids))

    def parse_dataset(self):

        logger.debug("parse_dataset")
        # 输出数据集引脚output1，本组件不对数据集做改动
        nodes = self.job.inputs[0].params.get('nodes')
        for node in nodes:
            nid = node.get('node').get('nid')
            if nid == self.curr_nid:
                self.ware_ctx.set_ware_output_data(OUTPUT, json.dumps(node))

        # 加载数据集为DataFrame
        if self.common_params.node_type == "NO_DATA_OWNER":
            self.data = None
            logger.debug("self.data: {}".format(self.data))
        else:
            self.data = self.load_dataset_input(self.dataset_input.url, self.dataset_input.content_type)
            logger.debug("self.data: {}".format(self.data.head(10)))

    def do_start(self):

        logger.debug("do_start")
        if self.first_run:
            self.flow_callback(self.job, "PAUSE")
        else:
            if len(self.selected_cols) > 0:
                self.pub, self.priv = paillier.generate_paillier_keypair(n_length=1024)
                self.num_data = self.data.shape[0]
                self.selected_data = self.data[self.selected_cols].add_prefix(self.nid[-4:]+'_')
                # 计算本地spearman系数矩阵local_matrix
                self.local_matrix = self.selected_data.corr(method="spearman")
                logger.debug("local_matrix: {}".format(self.local_matrix))
                # 计算本地rank，并加密
                self.rank = self.cal_rank()
                logger.debug("rank: {}".format(self.rank))
                self.rank_e = self.rank.applymap(self.pub.encrypt)
                logger.debug("rank_e: {}".format(self.rank_e))
                # 发送加密的rank给后面的节点
                for res_nid in self.next_nids:
                    logger.debug("self.nid: {} send rank_e to res_nid: {}".format(self.nid, res_nid))
                    self.algo_data_transfer.rank_e.send_by_nid(res_nid, self.rank_e, self.ctx, self.nid, res_nid, async_send=True)

                for pre_nid in self.prev_nids:
                    # 接收前面节点发来的加密的rank
                    logger.debug("from pre_nid: {} to self.nid: {} get rank_e".format(pre_nid, self.nid))
                    rank_e_pre = self.algo_data_transfer.rank_e.get(self.listener, pre_nid, self.nid)
                    logger.debug("rank_e_pre: {}".format(rank_e_pre.head(10)))
                    # 密态计算inter_matrix（对应每个前面节点）
                    inter_matrix_e_pre = rank_e_pre.T.dot(self.rank)
                    logger.debug("with pre_nid: {} inter_matrix_e_pre: {}".format(pre_nid, inter_matrix_e_pre))
                    # 发送密态的inter_matrix给每个前面的节点
                    self.algo_data_transfer.inter_matrix_e.send_by_nid(pre_nid, inter_matrix_e_pre, self.ctx, self.nid, pre_nid, async_send=True)

                self.inter_matrix_list = []
                for res_nid in self.next_nids:
                    # 接收后面每个节点发来的密态的inter_matrix
                    logger.debug("from res_nid: {} to self.nid: {} get inter_matrix_e".format(res_nid, self.nid))
                    inter_matrix_e_res = self.algo_data_transfer.inter_matrix_e.get(self.listener, res_nid, self.nid)
                    logger.debug("from res_nid: {} inter_matrix_e_res: {}".format(res_nid, inter_matrix_e_res))
                    # 解密inter_matrix
                    inter_matrix_res = inter_matrix_e_res.applymap(self.priv.decrypt)
                    inter_matrix_res = inter_matrix_res / self.num_data
                    logger.debug("from res_nid: {} inter_matrix_res: {}".format(res_nid, inter_matrix_res))
                    self.inter_matrix_list.append(inter_matrix_res)

                if not self.is_owner:
                    # 发送inter_matrix和local_matrix给owner
                    logger.debug("local_matrix: {}".format(self.local_matrix))
                    logger.debug("inter_matrix_list: {}".format(self.inter_matrix_list))
                    self.algo_data_transfer.assemble_matrix.send((self.local_matrix, self.inter_matrix_list), self.ctx, self.nid, async_send=True)
            else:
                if self.is_owner:
                    logger.warning("无数据发起方，不参与计算")
                else:
                    logger.warning("当前节点无被选中的特征，不参与计算")

            if self.is_owner:
                # 收集matrix
                if (self.data is None) or (len(self.selected_cols) == 0):
                    local_m_list = []
                    inter_m_list = []
                else:
                    local_m_list = [self.local_matrix]
                    inter_m_list = self.inter_matrix_list
                for nid in self.all_nids:
                    if nid != self.curr_nid:
                        logger.debug("from nid: {} to self.nid: {} get assemble_matrix".format(nid, self.nid))
                        local_m_res, inter_m_list_res = self.algo_data_transfer.assemble_matrix.get(self.listener, nid)
                        local_m_list.append(local_m_res)
                        inter_m_list += inter_m_list_res
                # 拼装total_matrix
                if len(local_m_list) > 1:
                    total_matrix = pd.concat(local_m_list)
                    for inter_m in inter_m_list:
                        total_matrix = total_matrix.fillna(inter_m).fillna(inter_m.T)
                else:
                    total_matrix = local_m_list[0]
                # 上报结果
                logger.debug("total_matrix: {}".format(total_matrix))
                remote_id = self.file_system_client.write_content(total_matrix.to_csv())
                result = {'spearman_matrix_remote_id': remote_id}
                self.ware_ctx.set_ware_data(DOWNLOAD_RESULT, json.dumps(result))

    def cal_rank(self):

        logger.debug("cal_rank")
        rank = self.selected_data.rank()
        rank = (rank - rank.mean())/rank.std(ddof=0)
        return rank