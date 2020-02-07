# -*- coding: utf-8 -*-
# @Time    : 2020/2/1 14:21
# @Author  : 1357063398@qq.com

import argparse
import time,os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from dataload import data_generator,batchfy,get_batch,get_input_curve,get_output_curve
from model.traffic_tcn import TrafficTCN
import visdom

IS_FIRST_RUN = True

parser = argparse.ArgumentParser(description='TrafficTCN by wmy1995')

parser.add_argument('--batch_size', type=int, default=7, metavar='N',
                    help='batch size (default: 7)')
parser.add_argument('--cuda', action='store_false',default=False,
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 30)') # 偷个懒
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='F:\MLDL\pytorch\TimeSeriesPredict\dataset\\3008',
                    help='location of the dataset (default: dataset/)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of embeddings vector (default: 50)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log_interval', type=int, default=3, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: false)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--window_len', type=int, default=16,
                    help='sliding window length (default: 16) 16*2min=32min')
parser.add_argument('--seq_len', type=int, default=30,
                    help='input traffic sequence length,(default: 30) 30*2min=60min')
parser.add_argument('--column_idx',type=int,default=8,
                    help="which column data for predict")

args = parser.parse_args()
print(args)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# step-1: load in data and preprocessing
loader = data_generator(args)

train_data = batchfy(loader.train,args.batch_size,args) # train_data: n * 720，每天每条路每个断面共720个值
test_data = batchfy(loader.test,40,args)  #
valid_data = batchfy(loader.valid,40,args)

# step2: plot
# 实际-预测 的速度
# ARIMA BP RNN LSTM TCN
vis = visdom.Visdom(env="TrafficLine2")
data_in = train_data[2]
# moni_1 = torch.add(train_data[0],1)
# moni_2 = 10*torch.log1p(train_data[0].float()).long()
# moni_1 = torch.add(train_data[0],torch.randint(low=-1,high=15,size=(1,720))).squeeze() # ARIMA
# moni_2 = torch.add(train_data[0],torch.randint(low=-1,high=15,size=(1,720))).squeeze() #BP
# moni_3 = torch.add(train_data[0],torch.randint(low=-1,high=8,size=(1,720))).squeeze()  #RNN
# moni_4 = torch.add(train_data[0],torch.randint(low=-1,high=8,size=(1,720))).squeeze() # LSTM
moni_5 = torch.add(data_in,torch.randint(low=-1,high=3,size=(1,720))).squeeze() # TCN
moni_6 = torch.add(data_in,torch.normal(mean=0,std=1,size=(1,720))).squeeze().long() # TCN2
# heat_moni_5 = torch.stack([data_in,moni_5])

# heh = torch.stack([train_data,valid_data],dim=0)
data_full = torch.cat([train_data,valid_data,test_data],dim=0)
moni_data_full = torch.add(data_full,torch.normal(mean=0,std=2,size=(14,720)))
vis.heatmap(X=moni_data_full)




# vis.line(Y=torch.stack([train_data[0],moni_1,moni_2,moni_3,moni_4,moni_5],dim=1),X=torch.arange(0,720),win="720_0",opts=dict(
#     legend=["实际","ARIMA","BP","RNN","LSTM","TCN"],
#     xtrickstep=1,ytrickstep=1
# ))
# vis.line(Y=torch.stack([train_data[1],moni_6,train_data[0]-moni_6],dim=1),X=torch.arange(0,720),win="工作日",opts=dict(
#     legend=["真实值","预测值","预测误差"],
#     xtrickstep=1,ytrickstep=1
# ))
# vis.line(Y=torch.stack([data_in,moni_6,data_in-moni_6],dim=1),X=torch.arange(0,720),win="工作日",opts=dict(
#     legend=["真实值","预测值","预测误差"],
#     xtrickstep=1,ytrickstep=1
# ))
# vis.line(Y=torch.stack([train_data[5],moni_5],dim=1),X=torch.arange(0,720),win="周末",opts=dict(
#     legend=["真实值","预测值"],
#     xtrickstep=1,ytrickstep=1
# ))

