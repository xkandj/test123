import json

import numpy as np

from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_score_card.score_card_transform_base import ScoreCardTransformBase

logger = get_fmpc_logger(__name__)


class ScoreCardTransformGuest(ScoreCardTransformBase):

    def do_ready(self) -> None:
        # 解析settings
        self.parse_settings(self.job.settings)
        # 检查model 参数， 校验iv woe，获取model里面的rules, 获取woe分箱值
        self.parse_woe_rules()
        # 获取model
        self.load_model()

    def do_start(self):
        # 计算新的model，weight
        self.log_info(f'开始进行评分卡转换...')
        model = self.calculate_score_weight()
        self.log_info(f'开始生成评分卡报告...')
        # 上报前端需要的
        report = self.generate_report()
        # 保存模型
        lr_score_report = {"lr_model": self.model, "score_model": model, "report": report}
        self.save_model(lr_score_report)
        self.log_info(f'保存评分卡模型...')

        # 上报报告
        if self.is_owner:
            self.update_model_report(report)
            logger.info("=======================执行保存资源操作开始===================")
            self.do_resource_save()

    def static_role(self) -> str:
        return 'GUEST'

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
            "score_max":567,
            "intercept_score": 324,  // 截距分，如果除了y其他节点没这个字段
            "score_table": [
                {
                    "name": "x1"， // 特征名称
                    "bins": ["-1.1,0", "0, 1.5",...], //分箱
                    "scores": [10,20,30,....] // 分值
                },
                {
                    "name": "x2"，
                    "bins": ["-1.1,0", "0, 1.5",...],
                    "scores": [10,20,30,....]
                },
                ...
            ]
        }
        """
        w, coef_ = self.model[0], self.model[3]
        intercept_score = self.A + self.B * coef_
        scores_table = []
        min_score = intercept_score
        max_score = intercept_score
        discrete_type = []
        for rule in self.rules:
            if rule.rule_type != 'woe':
                continue
            woe_list = json.loads(rule.content).get('woeList')
            name = rule.column_name
            if name not in w:
                continue
            scores_dict = {'name': name, 'bins': [], 'scores': [], 'featureMap': None, 'distributeType' : None}
            feature_min_score = np.Inf
            feature_max_score = -np.Inf
            discrete_feature_map = {}
            column_type = None
            for woe in woe_list:
                if woe.get('distributeType') == 'CONTINUOUS':
                    bin_min_ = str(woe.get('min')) if woe.get('min') else '-inf'
                    bin_max_ = str(woe.get('max')) if woe.get('max') else 'inf'
                    scores_dict['bins'].append(bin_min_ + ',' + bin_max_)
                    scores_dict['distributeType'] = 'CONTINUOUS'
                elif woe.get('distributeType') == 'DISCRETE':
                    scores_dict['bins'].append(woe.get('discre')[0])
                    scores_dict['distributeType'] = 'DISCRETE'
                    discrete_feature_map[woe.get('woe')] = woe.get('discre')[0]
                    if column_type is None:
                        discrete_type.append(woe.get('name'))
                        column_type = woe.get('name')
                else:
                    raise TypeError(f"无效的distributeType: {woe.get('distributeType')}")
                bin_score = self.B * w[name] * woe['woe']
                feature_min_score = min(feature_min_score, bin_score)
                feature_max_score = max(feature_max_score, bin_score)
                scores_dict['scores'].append(bin_score)
            scores_dict['featureMap'] = discrete_feature_map
            scores_table.append(scores_dict)
            min_score += feature_min_score
            max_score += feature_max_score
        report = {
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "A": self.A,
            "B": self.B,
            "score_min": min_score,
            "score_max": max_score,
            "intercept_score": intercept_score,
            "score_table": scores_table,
            "discrete_type": discrete_type
        }
        # if not self.is_owner:
        #     self.algo_data_transfer.min_max_score.send((min_score, max_score), self.ctx)
        self.algo_data_transfer.min_max_score.send((min_score, max_score), self.ctx)
        return report
