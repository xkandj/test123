import os
from fmpc.utils.EnvUtil import EnvUtil
from fmpc.utils import ConstUtil
from fmpc.web.AvatarApplication import registry
from fmpc.utils.LogUtils import get_fmpc_logger
from wares.hetero_score_card.score_card_transform_ware import ScoreCardTransformWare, score_card_app
from wares.hetero_score_card.score_card_evaluation_predict_ware import ScoreCardTransformEvaluationPredictWare, score_card_app_new

__logger = get_fmpc_logger(__name__)

if ConstUtil.ServerType.training in EnvUtil.get_server_type_list():
    __logger.info("ScoreCardTransformWare init")
    setting_file = os.path.join(os.path.dirname(__file__), "ware.json")
    registry(score_card_app, ScoreCardTransformWare(setting_file))
    __logger.info("ScoreCardTransformWare init end")

# TODO 03版本临时方案，暂时屏蔽注册在training上的所有预测组件，后续版本和批量预测、评估组件一起重新设计方案
# if ConstUtil.ServerType.api_predict in EnvUtil.get_server_type_list():
__logger.info('LRPredictWare init')
setting_file = os.path.join(os.path.dirname(__file__), 'ware_predict.json')
predict_ware = ScoreCardTransformEvaluationPredictWare(setting_file)
predict_ware.register_server_type = ConstUtil.ServerType.api_predict
registry(score_card_app_new, predict_ware)
__logger.info('LRPredictWare init end')

