from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_score_card.score_card_transform_base import ScoreCardTransformBase
from wares.common.fmpc_error import RoleError

logger = get_fmpc_logger(__name__)


class ScoreCardTransformNodata(ScoreCardTransformBase):

    def __init__(self, ware_id, **kwargs):
        super().__init__(ware_id, **kwargs)

    def do_ready(self) -> None:
        # 解析settings
        self.parse_settings(self.job.settings)

    def do_start(self):
        # 计算新的model，weight
        self.log_info(f'开始进行评分卡转换...')
        model = ()
        # 上报前端需要的
        report = self.generate_report()
        # 保存模型
        self.save_model(model)
        # 上报报告
        if not self.is_owner:
            raise RoleError('角色异常！无数据方必须是发起方！')
        self.update_model_report(report)
        self.log_info(f'保存评分卡模型...')
        if self.is_owner:
            logger.info("=======================执行保存资源操作开始===================")
            self.do_resource_save()

    def static_role(self) -> str:
        return 'NODATA'

    def generate_report(self) -> dict:
        """
        report 结构例子:
        {
            "base_score": 123,  // 基准分
            "base_odds": 213,   // 基准赔率
            "pdo": 123,    //
            "A": 123,
            "B": 123,
            "score_min":678,
            "score_max":567
        }
        """
        min_score, max_score = self.algo_data_transfer.min_max_score.get(self.listener)
        report = {
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "A": self.A,
            "B": self.B,
            "score_min": min_score,
            "score_max": max_score
        }
        return report
