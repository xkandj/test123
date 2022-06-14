import pandas as pd
import numpy as np
import copy
import random
from fmpc.utils.LogUtils import get_fmpc_logger

logger = get_fmpc_logger(__name__)


class BinProcesser:
    def __init__(self,
                 data_arr=None,
                 label_arr=None,
                 is_label=False,
                 bin_method=None,
                 bins_param={},
                 feature_type='continuous',
                 role='guest',
                 transfer=None):
        """
        data_arr: 分箱数据
        label_arr: 标签值
        is_label: bool {True, False}处理的特征是否为标签值
        bin_method: 分箱方式{'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bins_param: dict 分箱参数
                  'num_bins'：分箱的数
                  'min_sample_num': 每个分箱中的最小样本数
                  'chi_threshold': 卡方阈值  只有使用卡方分箱时才会用到这个变量
                  'bins_list_custom': list 自定义时的分箱列表 如:连续[1,3]; 离散[['a'],['b','c']]
        feature_type: 特征类型,{'continuous','categorical'}
        role:  角色
        transfer: 通信函数
        """
        self.data_arr = data_arr
        self.label_arr = label_arr
        self.is_label = is_label
        self.bin_method = bin_method
        self.feature_type = feature_type
        self.role = role
        self.transfer = transfer
        if self.bin_method in ['distince_bin', 'frequency_bin', 'chimerge_bin']:
            self.num_bins = bins_param.get('num_bins')
            if self.num_bins is None:
                raise ValueError('请确保等频，等距，卡方分箱中含有参数分箱数')
        if self.bin_method in ['distince_bin', 'chimerge_bin']:
            self.min_sample_num = bins_param.get('min_sample_num')
            if self.num_bins is None:
                raise ValueError('请确保等距，卡方分箱中含有参数分箱中最小样本数')
        if self.bin_method in ['chimerge_bin']:
            self.chi_threshold = bins_param.get('chi_threshold')
            if self.num_bins is None:
                raise ValueError('请确保卡方分箱中含有参数卡方值')
        if self.bin_method in ['custom_bin']:
            self.bins_list_custom = bins_param.get('bins_list_custom')
            if self.bins_list_custom is None:
                raise ValueError('请确保自定义分箱中含有参数分箱列表')

    def check_bins_list(self, bins_list_arr):
        """
        设置连续特征分箱的两端的分裂点，并返回list类型的分箱信息
        bins_list_arr: array, 连续特征的分箱边界值
        return:
           bins_list: list 连续特征的分箱边界值
        """
        if self.is_label:
            bins_list = [-0.0001] + bins_list_arr.tolist()[1:-1] + [1]
        else:
            bins_list = [float('-inf')] + list(bins_list_arr)[1:-1] + [float('inf')]
        return bins_list

    def check_bins_list_custom(self, custom_bins_list):
        """
        对自定义的分箱列表，设置连续特征分箱的两端的分裂点
        custom_bins_list: 自定义的分箱列表 如[1,2]
        return:
            bins_list: list 连续特征的分箱边界值
        """
        if self.is_label:
            bins_list = [-0.0001] + custom_bins_list[1:-1] + [1]
        else:
            bins_list = [float('-inf')] + custom_bins_list[1:-1] + [float('inf')]
        return bins_list

    def get_chi_merge_bins_continuous(self, data_arr, label_arr):
        """获取特征中每个元素或每个分箱的正负样本数，和分箱边界的list"""
        # 获取特征元素对应的 标签为0的样本个数，标签为1的样本个数
        col_0 = []
        col_1 = []
        data_list = list(set(data_arr))
        data_list.sort()
        tmp_ = 0
        for each_data in data_list:
            index_ = data_arr == each_data
            label_ = label_arr[index_]
            for i in label_:
                tmp_ += i
            col_1.append(tmp_)
            col_0.append(index_.sum() - tmp_)

        data_ = {float(0): col_0, float(1): col_1}
        freq_tab = pd.DataFrame(data_, index=data_list)
        # 在密态的情况下，无法使用pd.crosstab这个方法
        # freq_tab = pd.crosstab(data_arr, label_arr)
        # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.
        bins_list_arr = freq_tab.index.values
        # bins_list 分箱的列表，包含分箱的上边界和下边界
        bins_list1 = list(bins_list_arr)
        bins_list = [float('-inf')] + bins_list1[:-1] + [float('inf')]

        if len(bins_list) > 50:  # 应该是50+1吧，有范围的两端呀
            # 优化方案：等频法预分箱50箱
            _, bins_list_arr = pd.qcut(data_arr, 50, duplicates='drop', retbins=True)
            bins_list = self.check_bins_list(bins_list_arr)

            bad_p = []
            freq = []
            for i in range(1, len(bins_list)):
                index_tmp = (data_arr > bins_list[i - 1]) & (data_arr <= bins_list[i])
                if len(index_tmp) > 0:
                    count1 = label_arr[index_tmp].sum()  # 分箱中的正样本的个数
                    count0 = len(index_tmp) - count1  # 分箱中的负样本的个数
                    freq.append([count0, count1])  # 从小->大
                else:
                    if i != float('inf'):
                        bad_p.append(bins_list[i])  # 针对分割区域无样本的情况，如100样本，95个-1，等频分割10箱
                    continue
            if len(bad_p) > 0:
                for p in bad_p:
                    bins_list.remove(p)
        else:
            freq = freq_tab.values  # 将原来的dataFrame变成了array类型
        return np.array(freq), bins_list

    def chi_val(self, arr):
        """
        计算卡方值
        :param arr: 频数统计表,二维array
        return:
          chi_square_value: 卡方值
          tn: 两个分箱中的所有样本数
        """
        assert (arr.ndim == 2)  # arr.ndim返回数组的维度。 判断数组维度
        rn = arr.sum(axis=1)  # 计算每行总频数  即每个分箱中的样本数
        cn = arr.sum(axis=0)  # 每列总频数     正样本总的个数，负样本总的个数
        tn = arr.sum()  # 总频数        所有的样本数

        exp = cn.T * rn / tn
        exp_ = copy.deepcopy(exp)
        exp_[exp_ == 0] = 1
        square = (arr - exp) ** 2 / exp_
        # 卡方值: sum((arr-exp)**2 / exp)
        chi_square_value = square.sum()
        return chi_square_value, tn

    def chi_merge_for_continuous(self):
        """  需要考虑分箱中的样本数判断条件
        卡方分箱 连续特征
        """
        # 获取卡方分箱的每个分箱中正负样本的个数，及分箱边界值的list
        # freq ：array； bins_list：list
        logger.info('======>>>>>> 初始化卡方分箱 con')
        freq, bins_list = self.get_chi_merge_bins_continuous(self.data_arr, self.label_arr)
        # 优化合并分箱
        logger.info('======>>>>>> 合并卡方分箱开始 con')
        while True:
            min_chi = None
            min_idx = None
            min_samp = None
            # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
            logger.info('======>>>>>> 遍历所有分箱，寻找最小卡方值对应的分箱 con')
            for i in range(len(freq) - 1):  # 遍历每个分箱
                v, tn = self.chi_val(freq[i:i + 2])  # 计算卡方值  相连的两个分箱
                if min_chi is None or (min_chi > v):  # 小于当前最小卡方，更新最小值
                    min_chi = v
                    min_idx = i
                    min_samp = tn
            # 如果最小卡方值小于阈值，则自下而上合并最小卡方值的相邻两组，并继续循环
            logger.debug("->, freq.sum={}, len(freq)={}, min_chi:{}".format(freq.sum(), len(freq), min_chi))
            # if (bins < len(freq) and min_chi < threshold) or min_samp < min_sample_num:
            logger.info('======>>>>>> 判断最小卡方值的分箱是否满足合并条件 con')
            if len(freq) > self.num_bins or min_chi < self.chi_threshold:
                tmp = freq[min_idx] + freq[min_idx + 1]
                freq[min_idx] = tmp
                freq = np.delete(freq, min_idx + 1, 0)  # 删除min_idx后一行
                # 删除当前min_idx对应的切分点，保留min_idx+1对应的切分点，其中freq中的第i个分箱的上界值对应与bin_list中第i+1个值
                bins_list.pop(min_idx + 1)
            # 最小卡方值不小于阈值，停止合并。
            else:
                break
        logger.info('======>>>>>> 合并卡方分箱完成 con')
        return bins_list

    def get_bins_continuous(self, transfer_param=None):
        """
        获取分箱list  连续变量
        transfer_param : {"ctx": self.ctx, "listener": self.listener}
        """
        # 异常处理: 判断特征列是否是常量
        if np.min(self.data_arr) == np.max(self.data_arr):
            raise ValueError("PSI: 目标特征的取值唯一，即常量，无法计算PSI.")
        if self.bin_method == "distince_bin":
            _, bins_list_arr = pd.cut(self.data_arr, self.num_bins, duplicates='drop', retbins=True)
            # 看一下是否有最小分箱样本数，如果有，需要合并分箱
            # 更改bins_list两端的边界值
            bins_list = self.check_bins_list(bins_list_arr)
        elif self.bin_method == "frequency_bin":
            _, bins_list_arr = pd.qcut(self.data_arr, self.num_bins, duplicates='drop', retbins=True)
            # 看一下是否有最小分箱样本数，如果有，需要合并分箱
            # 更改bins_list两端的边界值
            bins_list = self.check_bins_list(bins_list_arr)
        elif self.bin_method == "chimerge_bin":
            # 分箱数，最小卡方阈值，分箱最少样本数
            if self.role == 'guest':
                bins_list = self.chi_merge_for_continuous()
            elif self.role == 'host':
                bins_list = self.chi_merge_for_continuous_host(transfer_param)
        elif self.bin_method == "custom_bin":
            # 需要根据具体需求进行书写
            bins_list = self.check_bins_list_custom(self.bins_list_custom)
        return bins_list

    def decrypted_cal_chi_val(self, freq_dict, priv_decrypt):
        """ 计算host节点最小卡方值 """
        min_chi = None
        min_idx = None
        min_samp = None
        # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
        for i, freq_arr in freq_dict.items():
            assert (freq_arr.ndim == 2)  # arr.ndim返回数组的维度。 判断数组维度
            arr = np.zeros((2, 2), np.int)
            # 解密
            arr[0, 0] = round(priv_decrypt(freq_arr[0, 0]), 0)
            arr[0, 1] = round(priv_decrypt(freq_arr[0, 1]), 0)
            arr[1, 0] = round(priv_decrypt(freq_arr[1, 0]), 0)
            arr[1, 1] = round(priv_decrypt(freq_arr[1, 1]), 0)

            v, tn = self.chi_val(arr)
            if min_chi is None or (min_chi > v):  # 小于当前最小卡方，更新最小值
                min_chi = v
                min_idx = i
                min_samp = tn
        return min_idx, min_chi, min_samp

    def chi_merge_for_con_or_cat_guest(self, transfer_param, priv_decrypt):
        """
        卡方分箱中，需要处理的特征是host方时，需要两方进行交互最终得到卡方分箱
        guest方配合host的操作
        """
        # 解析通信参数
        ctx_transfer = transfer_param.get('ctx')
        listener_transfer = transfer_param.get('listener')
        is_do = True
        count = 0
        while is_do:
            count += 1
            # 获取host发送给guest方的相连两个分箱的密态样本统计数
            freq_dict_guest = self.transfer.chi_val_dict.get(listener_transfer, count)
            min_idx, min_chi, min_samp = self.decrypted_cal_chi_val(freq_dict_guest, priv_decrypt)
            chi_value = {'min_idx': min_idx, 'min_chi': min_chi, 'min_samp': min_samp}
            # 发送计算的最小卡方值及相关信息
            self.transfer.min_chi_val.send(chi_value, ctx_transfer, count)
            # 获取host方发送的是否运行的状态
            is_do = self.transfer.is_do.get(listener_transfer, count)

    def chi_merge_for_continuous_host(self, transfer_param):
        """     需要考虑分箱中的样本数判断条件
           卡方分箱 连续特征
           transfer_param:{"ctx": self.ctx, "listener": self.listener}
        """
        # 获取卡方分箱的每个分箱中正负样本的个数，及分箱边界值的list
        # freq ：array； bins_list：list
        logger.info('======>>>>>> 初始化卡方分箱 con')
        freq, bins_list = self.get_chi_merge_bins_continuous(self.data_arr, self.label_arr)

        # 解析通信参数
        ctx_transfer = transfer_param.get('ctx')
        listener_transfer = transfer_param.get('listener')
        logger.info('======>>>>>> 合并卡方分箱进行中')
        count = 0
        while True:
            count += 1
            freq_dict_guest = {}
            freq_list = list(range(len(freq) - 1))
            random.seed(count)
            freq_list = random.sample(freq_list, len(freq_list))

            for t, f in enumerate(freq_list):
                freq_dict_guest[f] = freq[t:t + 2]
            # 发送各分箱正负样本对个数
            logger.info('======>>>>>> 发送各分段数据->guest')
            self.transfer.chi_val_dict.send(freq_dict_guest, ctx_transfer, count)
            logger.info('======>>>>>> 获取guest方计算的最小卡方值')
            chi_res = self.transfer.min_chi_val.get(listener_transfer, count)

            min_idx_guest = chi_res.get('min_idx')
            min_chi = chi_res.get('min_chi')
            min_samp = chi_res.get('min_samp')

            min_idx = list(freq_dict_guest.keys()).index(min_idx_guest)
            # 如果最小卡方值小于阈值，则自下而上合并最小卡方值的相邻两组，并继续循环
            # if (bins < len(freq) and min_chi < threshold) or min_samp < min_sample_num:
            if len(freq) > self.num_bins or min_chi < self.chi_threshold:
                tmp = freq[min_idx] + freq[min_idx + 1]
                freq[min_idx] = tmp
                freq = np.delete(freq, min_idx + 1, 0)  # 删除min_idx后一行
                bins_list.pop(min_idx + 1)  # 删除对应的切分点
                is_do = True
            # 最小卡方值不小于阈值，停止合并。
            else:
                is_do = False
            # 发送是否继续进行卡方分箱计算
            self.transfer.is_do.send(is_do, ctx_transfer, count)
            if not is_do:
                break
        return bins_list

    def get_chi_merge_bins_categorical(self, data_arr, label_arr):
        """获取特征中每个元素或每个分箱的正负样本数，和分箱  离散特征"""
        # 离散特征中的元素
        bins_list = pd.Series(data_arr).unique().tolist()
        # 每个离散元素正样本的个数，负样本的个数
        freq = []
        for index, point in enumerate(bins_list):
            index_tmp = data_arr == point
            count1 = label_arr[index_tmp].sum()
            count0 = len(index_tmp) - count1
            freq.append([count0, count1])
        freq = np.array(freq)
        return freq, bins_list

    def chi_merge_for_categorical(self):
        """
        获取离散型变量卡方分箱点
        return: 包括各组的起始值的列表
        """
        logger.info('======>>>>>> 初始化卡方分箱 cate')
        freq, bins_list = self.get_chi_merge_bins_categorical(self.data_arr, self.label_arr)

        # 每一个离散元素单独一箱
        res = {v: [v] for v in bins_list}

        while True:
            min_chi = None
            min_idx = None
            min_samp = None

            # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
            for i in range(len(freq) - 1):
                v, tn = self.chi_val(freq[i:i + 2])  # 计算卡方值
                if min_chi is None or (min_chi > v):  # 小于当前最小卡方，更新最小值
                    min_chi = v
                    min_idx = i
                    min_samp = tn
            # 如果最小卡方值小于阈值，则自下而上合并最小卡方值的相邻两组，并继续循环
            if len(freq) > self.num_bins or min_samp < self.min_sample_num:
                # if (len(freq) > num_bins and min_chi < chi_threshold) or min_samp < min_sample_num:
                # 合并min_idx+1行到min_idx行
                res[bins_list[min_idx]].extend(res[bins_list[min_idx + 1]])
                res.pop(bins_list[min_idx + 1])

                tmp = freq[min_idx] + freq[min_idx + 1]
                # 替换为合并后的值
                freq[min_idx] = tmp
                # 删除min_idx后一行
                freq = np.delete(freq, min_idx + 1, 0)
                # 删除对应的切分点
                bins_list.pop(min_idx + 1)

                if len(freq) < 2 and freq.sum() == len(self.data_arr):
                    return list(res.values())
            # 最小卡方值不小于阈值，停止合并。
            else:
                break
        return list(res.values())

    def chi_merge_for_categorical_host(self, transfer_param):
        """  获取离散型变量卡方分箱点 需要与guest方进行交互 """
        logger.info('======>>>>>> 初始化卡方分箱 cate')
        freq, bins_list = self.get_chi_merge_bins_categorical(self.data_arr, self.label_arr)
        # 解析通信参数
        ctx_transfer = transfer_param.get('ctx')
        listener_transfer = transfer_param.get('listener')
        # 每一个离散元素单独一箱
        res = {v: [v] for v in bins_list}

        count = 0
        while True:
            count += 1
            freq_dict_guest = {}
            freq_list = list(range(len(freq) - 1))
            random.seed(count)
            freq_list = random.sample(freq_list, len(freq_list))

            for t, f in enumerate(freq_list):
                freq_dict_guest[f] = freq[t:t + 2]
            # 发送各分段数据->guest
            self.transfer.chi_val_dict.send(freq_dict_guest, ctx_transfer, count)
            # 获取guest方计算的最小卡方值
            chi_res = self.transfer.min_chi_val.get(listener_transfer, count)
            min_idx_guest = chi_res.get('min_idx')
            min_chi = chi_res.get('min_chi')
            min_samp = chi_res.get('min_samp')

            min_idx = list(freq_dict_guest.keys()).index(min_idx_guest)
            # 如果最小卡方值小于阈值，则自下而上合并最小卡方值的相邻两组，并继续循环
            if self.num_bins < len(freq) or min_chi < self.chi_threshold:  # or min_samp < min_sample_num:
                # res[cutoffs[min_idx]].append(cutoffs[min_idx + 1])     # 合并min_idx+1行到min_idx行
                res[bins_list[min_idx]].extend(res[bins_list[min_idx + 1]])
                res.pop(bins_list[min_idx + 1])
                tmp = freq[min_idx] + freq[min_idx + 1]
                freq[min_idx] = tmp  # 替换为合并后的值
                freq = np.delete(freq, min_idx + 1, 0)  # 删除min_idx后一行
                bins_list.pop(min_idx + 1)  # 删除对应的切分点. todo: 注: np.delete有坑,会将数值型转字符串

                is_do = True if len(freq) < 2 else False
            # 最小卡方值不小于阈值，停止合并。
            else:
                is_do = False
            self.transfer.is_do.send(is_do, ctx_transfer, count)
            if not is_do:
                break
        return list(res.values())

    def get_bins_categorical(self, transfer_param=None):
        """
        计算离散特征的分箱
        transfer_param: 通信参数
        return: bins_list 返回分箱的list
        """
        # 异常处理: 判断特征列是否是常量
        if len(list(set(self.data_arr))) == 1:
            raise ValueError("PSI: 目标特征的取值唯一，即常量，无法计算PSI.")
        if self.bin_method == 'discre_enum_bin':  # 默认分箱：枚举，每种取值为一箱
            bins_tem = np.unique(self.data_arr)
            bins_list = [[i] for i in bins_tem]
        elif self.bin_method == "chimerge_bin":
            # 分箱数，最小卡方阈值，分箱最少样本数
            if self.role == 'guest':
                bins_list = self.chi_merge_for_categorical()
            elif self.role == 'host':
                bins_list = self.chi_merge_for_categorical_host(transfer_param)
        elif self.bin_method == "custom_bin":
            bins_list = self.bins_list_custom
        return bins_list

    def get_bins_list(self, transfer_param=None):
        """ 获得分箱list """
        if self.feature_type == 'continuous':
            logger.info('======>>>>>> 待处理的特征为连续型特征，开始计算bins_list')
            bins_list = self.get_bins_continuous(transfer_param)
        elif self.feature_type == 'categorical':
            logger.info('======>>>>>> 待处理的特征为离散型特征，开始计算bins_list')
            bins_list = self.get_bins_categorical(transfer_param)
        return bins_list
