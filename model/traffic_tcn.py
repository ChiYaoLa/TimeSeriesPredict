# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 14:51
# @Author  : wmy1995

import torch
from torch import nn
import sys
from .tcn import TemporalConvNet

class TrafficTCN(nn.Module):
    def __init__(self,emb_size,n_categs,channels_size,
                 kernel_size=2,dropout=0.3,emb_dropout=0.1,tied_weights=False):
        super(TrafficTCN,self).__init__()

        self.encoder = nn.Embedding(n_categs,emb_size)
        self.traffic_tcn = TemporalConvNet(emb_size,channels_size,kernel_size,dropout=dropout)
        self.decoder = nn.Linear(channels_size[-1],n_categs)  #卷积的输出channel 接着 n_categs
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()  # init weights

    def init_weights(self):
        self.encoder.weight.data.normal_(0,0.01)
        self.decoder.weight.data.normal_(0,0.01)
        self.decoder.bias.data.fill_(0)

    def forward(self, input):
        """
        input: n * sequence_len (sequence_len:你准备用多长的序列预测下一个值，预先可以配置)
        emb: n * sequence_len * emb_size
        """
        # input = input.to(torch.int64)
        # input = self.encoder(input.to(torch.int64))*1000
        emb = self.drop(self.encoder(input.to(torch.int64))*1000)
        """
        emb.transpose(1,2): n * emb_size * sequence_len
        y :                 n *  sequence_len * channes[-1]
        """
        y = self.traffic_tcn(emb.transpose(1,2)).transpose(1,2)
        """
        y: n *  sequence_len * n_cates
        """
        y = self.decoder(y)  # 如果我把这些整数值 全看成类别 不就简单了

        return y.contiguous()




