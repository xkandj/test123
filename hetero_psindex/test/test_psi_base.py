import numpy as np
import unittest
from hetero_psindex.hetero_psindex_base import get_bins_cnt_range


class TestCases(unittest.TestCase):
    def setUp(self) -> None:
        super(TestCases, self).setUp()

    def tearDown(self) -> None:
        ...

    def test_get_bins_cnt_range_0(self):
        """对连续变量 使用分箱值，统计data_arr在每个分箱中的样本数"""
        data_arr = np.array([0, 1, 2, 3, 4, 5, 6, -1])
        bins = [-float('inf'), 1, 5, float('inf'), 'NaN']
        data_type = 'continuous'
        cnt_array, score_range_list = get_bins_cnt_range(data_arr, bins, data_type)
        cnt_array_result = np.array([3, 4, 1, 0])
        score_range_list_result = ['(-inf,1]', '(1,5]', '(5,inf)', 'NaN']
        is_right1 = (cnt_array == cnt_array_result).all()
        is_right2 = score_range_list == score_range_list_result
        assert (is_right1 & is_right2)

    def test_get_bins_cnt_range_1(self):
        """对离散变量 使用分箱值，统计data_arr在每个分箱中的样本数"""
        data_arr = np.array(['1', '1', '2', '', '4', '5', '2', 'NaN'])
        bins = [['1', '2'], ['4', '5'], ['NaN']]
        data_type = 'categorical'
        cnt_array, score_range_list = get_bins_cnt_range(data_arr, bins, data_type)
        cnt_array_result = np.array([4, 2, 2])
        score_range_list_result = ["['1', '2']", "['4', '5']", "['NaN']"]
        is_right1 = (cnt_array == cnt_array_result).all()
        is_right2 = score_range_list == score_range_list_result
        assert (is_right1 & is_right2)


if __name__ == "__main__":
    unittest.main()
