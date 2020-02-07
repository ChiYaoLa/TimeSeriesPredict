# -*- coding: utf-8 -*-
# @Time    : 2020/2/7 11:17
# @Author  : 1357063398@qq.com

import unittest
from model.periodic_factor import PeriodicFactor
import torch

class TestPeriodicFactor(unittest.TestCase):
    def setUp(self):
        self.mock_data_1 = torch.ones(size=(3,720))
        self.mock_data_2 = torch.randint(high=10,size=(3,3)).float() # 因为mean只能计算float类型
        self.mock_sample = torch.randint(high=8,size=(2,3)).float()
        self.pre_fac = PeriodicFactor(self.mock_data_2)

    def test_factors(self):
        self.pre_fac._factors()
        print(self.pre_fac.mean_val)
        # self.assertTrue(torch.eq(self.pre_fac.mean_val,torch.ones(size=(1,3))))

    def test_predict(self):
        self.pre_fac.predict(self.mock_sample)
        # ok。。

if __name__ == "__main__":
    unittest.main()