# -*- coding: utf-8 -*-
# @Time    : 2020/2/7 10:31
# @Author  : 1357063398@qq.com
# 周期因子法


import torch

class PeriodicFactor:
    """data : n*720 tensor"""
    def __init__(self,data):
        self.data = data.float() # 输入数据
        self.mean_val = None # 均值
        self.mid_val = None # 中位数,最终的周期因子
        self.out = None # 预测数据

    def _factors(self):
        self.mean_val = torch.mean(self.data,dim=1).squeeze()
        for i in range(self.data.size(0)):
            self.data[i] = torch.div(self.data[i],self.mean_val[i])
        self.mid_val = torch.median(self.data,dim=0).values
        return  self.mid_val

    def predict(self,sample):
        """sample 一维的"""
        self._factors()
        sample = sample.float()
        sam_base = torch.mean(sample)
        self.out = self.mid_val*sam_base
        # self.out = torch.matmul(sam_base.unsqueeze(dim=1),self.mid_val.unsqueeze(dim=0))
        return self.out



