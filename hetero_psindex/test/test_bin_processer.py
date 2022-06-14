import numpy as np
from wares.hetero_psindex.bin_processer import BinProcesser
import unittest


class TestCases(unittest.TestCase):
    def setUp(self) -> None:
        super(TestCases, self).setUp()

    def tearDown(self) -> None:
        ...

    def test_get_bins_continuous_01(self):
        """ 连续特征 等距分箱"""
        data_arr1 = np.array(
            [-1.056642387, -3.477615208, 8.658658588, 2.06125457, -1.22852772, -4.4460423, -5.493267668, 2.469199308,
             2.700538452, -2.014950189, 7.078827721, -2.77853244, -3.490189392, -2.272462631, -2.990915071,
             6.247391444, 1.737421244, 8.855132932, 5.802243221, -0.395869944, -0.954921526, -4.403204379,
             0.604928043, 6.417668543, 0.381925943])
        data_arr2 = np.array(
            ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
             "b", "b", "a", "b"])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'distince_bin'
        bins_param = {'num_bins': 4, 'min_sample_num': 2}

        bin_processer_c = BinProcesser(
            data_arr=data_arr1,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            bins_param=bins_param,
            feature_type='continuous',
            role='guest',
            transfer=None)
        bins_list = bin_processer_c.get_bins_continuous()
        bins_list_result = [float('-inf'), -1.9061675179999997, 1.6809326320000002, 5.268032782000001, float('inf')]
        is_right = bins_list == bins_list_result
        print("测试用例一：连续特征  等距分箱")
        print(is_right)
        assert (is_right)

    def test_get_bins_continuous_02(self):
        """ 连续特征 等频分箱"""
        data_arr1 = np.array(
            [-1.056642387, -3.477615208, 8.658658588, 2.06125457, -1.22852772, -4.4460423, -5.493267668, 2.469199308,
             2.700538452, -2.014950189, 7.078827721, -2.77853244, -3.490189392, -2.272462631, -2.990915071,
             6.247391444, 1.737421244, 8.855132932, 5.802243221, -0.395869944, -0.954921526, -4.403204379,
             0.604928043, 6.417668543, 0.381925943])
        data_arr2 = np.array(
            ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
             "b", "b", "a", "b"])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'frequency_bin'
        bins_param = {'num_bins': 4}

        bin_processer_c = BinProcesser(
            data_arr=data_arr1,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            bins_param=bins_param,
            feature_type='continuous',
            role='guest',
            transfer=None)
        bins_list = bin_processer_c.get_bins_continuous()
        bins_list_result = [float('-inf'), -2.77853244, -0.395869944, 2.700538452, float('inf')]
        is_right = bins_list == bins_list_result
        print("测试用例二：连续特征  等频分箱")
        print(is_right)
        assert (is_right)

    def test_get_bins_continuous_03(self):
        """ 连续特征 卡方分箱"""
        data_arr1 = np.array(
            [-1.056642387, -3.477615208, 8.658658588, 2.06125457, -1.22852772, -4.4460423, -5.493267668, 2.469199308,
             2.700538452, -2.014950189, 7.078827721, -2.77853244, -3.490189392, -2.272462631, -2.990915071,
             6.247391444, 1.737421244, 8.855132932, 5.802243221, -0.395869944, -0.954921526, -4.403204379,
             0.604928043, 6.417668543, 0.381925943])
        data_arr2 = np.array(
            ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
             "b", "b", "a", "b"])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'chimerge_bin'
        bins_param = {'num_bins': 4, 'min_sample_num': 2, 'chi_threshold': 3.841}

        bin_processer_c = BinProcesser(
            data_arr=data_arr1,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            bins_param=bins_param,
            feature_type='continuous',
            role='guest',
            transfer=None)
        bins_list = bin_processer_c.get_bins_continuous()
        bins_list_result = [float('-inf'), 0.381925943, 2.469199308, 2.700538452, float('inf')]
        is_right = bins_list == bins_list_result
        print("测试用例三：连续特征 卡方分箱")
        print(is_right)
        assert (is_right)

    def test_get_bins_categorical_01(self):
        """ 离散特征 单个元素成一箱"""
        data_arr2 = np.array(
            ["a", "a", "a", "a", "c", "a", "d", "a", "e", "a", "c", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
             "b", "b", "a", "b"])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        is_label = False
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'discre_enum_bin'

        bin_processer_c = BinProcesser(
            data_arr=data_arr2,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            feature_type='categorical',
            role='guest',
            transfer=None)
        bins_list = bin_processer_c.get_bins_categorical()
        bins_list_result = [['a'], ['b'], ['c'], ['d'], ['e']]
        is_right = bins_list == bins_list_result
        print("测试用例一：离散特征  discre_enum_bin")
        print(is_right)
        assert (is_right)

    def test_get_bins_categorical_02(self):
        """ 离散特征 卡方分箱"""
        data_arr2 = np.array(
            ["a", "a", "a", "a", "c", "a", "d", "a", "e", "a", "c", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a",
             "b", "b", "a", "b"])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'chimerge_bin'
        bins_param = {'num_bins': 4, 'min_sample_num': 2, 'chi_threshold': 3.841}
        bin_processer_c = BinProcesser(
            data_arr=data_arr2,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            bins_param=bins_param,
            feature_type='categorical')
        bins_list = bin_processer_c.get_bins_categorical()
        bins_list_result = [['a'], ['c', 'd'], ['e'], ['b']]
        is_right = bins_list == bins_list_result
        print("测试用例二：离散特征  卡方分箱")
        print(is_right)
        assert (is_right)

    def test_get_bins_categorical_03(self):
        """ 离散特征 卡方分箱  元素为数值型"""
        data_arr2 = np.array([1, 1, 1, 1, 2, 1, 5, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        is_label = False
        # {'distince_bin','frequency_bin','chimerge_bin','discre_enum_bin','custom_bin'}
        bin_method = 'chimerge_bin'
        bins_param = {'num_bins': 4, 'min_sample_num': 2, 'chi_threshold': 3.841}
        bin_processer_c = BinProcesser(
            data_arr=data_arr2,
            label_arr=label_arr,
            is_label=False,
            bin_method=bin_method,
            bins_param=bins_param,
            feature_type='categorical')
        bins_list = bin_processer_c.get_bins_categorical()
        bins_list_result = [[1], [2, 5], [4], [3]]
        is_right = bins_list == bins_list_result
        print("测试用例三：离散特征  卡方分箱 数值型")
        print(is_right)
        assert (is_right)

    def test_chi_val(self):
        """ 计算卡方值 """
        bin_processer_c = BinProcesser()
        arr = np.array([[1, 2], [3, 4]])
        # 返回卡方值和总的样本数
        chi_square_value, tn = bin_processer_c.chi_val(arr)
        chi_square_value_result = 3.8952380952380956
        tn_result = 10
        is_right = (chi_square_value_result == chi_square_value) & (tn == tn_result)
        print("计算卡方值")
        print(is_right)
        assert (is_right)

    def test_get_chi_merge_bins_categorical(self):
        """卡方分箱的初始分箱  离散特征"""
        data_arr2 = np.array([1, 1, 1, 1, 2, 1, 5, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        bin_processer_c = BinProcesser()
        freq, bins_list = bin_processer_c.get_chi_merge_bins_categorical(data_arr2, label_arr)
        freq_result = np.array([[16, 9], [25, 0], [25, 0], [24, 1], [23, 2]])
        bins_list_result = [1, 2, 5, 4, 3]
        is_right = (freq == freq_result).all() & (bins_list == bins_list_result)
        print("卡方分箱的初始分箱  离散特征")
        print(is_right)
        assert (is_right)

    def test_get_chi_merge_bins_continuous(self):
        """ 卡方分箱的初始分箱 连续特征"""
        data_arr1 = np.array(
            [-1.056642387, -3.477615208, 8.658658588, 2.06125457, -1.22852772, -1.22852772, -4.403204379, 2.469199308,
             2.700538452, -1.056642387, -1.22852772, -1.056642387, -3.490189392, -4.403204379, -2.990915071,
             6.247391444, 2.06125457, 8.855132932, 5.802243221, -0.395869944, -4.403204379, -4.403204379,
             -1.22852772, 6.417668543, 2.06125457])
        label_arr = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        bin_processer_c = BinProcesser()
        freq, bins_list = bin_processer_c.get_chi_merge_bins_continuous(data_arr1, label_arr)
        freq_result = np.array(
            [[2, 2], [0, 1], [0, 1], [0, 1], [3, 1], [2, 1], [1, 0], [1, 2], [0, 1], [0, 1], [1, 0], [1, 0],
             [1, 0], [0, 1], [1, 0]])
        bins_list_result = [float("-inf"), -4.403204379, -3.490189392, -3.477615208, -2.990915071, -1.22852772,
                            -1.056642387, -0.395869944, 2.06125457, 2.469199308, 2.700538452, 5.802243221, 6.247391444,
                            6.417668543, 8.658658588, float("inf")]

        is_right = ((freq == freq_result).all()) & (bins_list == bins_list_result)
        print("卡方分箱的初始分箱  连续特征")
        print(is_right)
        assert (is_right)


if __name__ == "__main__":
    unittest.main()
