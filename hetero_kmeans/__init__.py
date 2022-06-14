# coding: utf-8
import os

from fmpc.web.AvatarApplication import registry
from wares.hetero_kmeans.k_means_predict_ware import k_means_predict_app, KMeansPredictWare
from wares.hetero_kmeans.k_means_ware import KMeansWare, kmeans_app
from fmpc.utils.LogUtils import get_fmpc_logger

# train
__logger = get_fmpc_logger(__name__)
__logger.info('KMeansWare init')
setting_file = os.path.join(os.path.dirname(__file__), 'ware.json')
registry(kmeans_app, KMeansWare(setting_file))
__logger.info('KMeansWare init end')

__logger.info('KMeansPredictWare init')
setting_file = os.path.join(os.path.dirname(__file__), 'ware_predict.json')
registry(k_means_predict_app, KMeansPredictWare(setting_file))
__logger.info('KMeansPredictWare init end')