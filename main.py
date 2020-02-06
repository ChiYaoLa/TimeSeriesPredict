# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 14:50
# @Author  : wmy1995

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
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit (default: 30)') # 偷个懒
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='F:\MLDL\pytorch\TimeSeriesPredict\dataset\\3005',
                    help='location of the dataset (default: dataset/)')
parser.add_argument('--emsize', type=int, default=70,
                    help='size of embeddings vector (default: 70)')
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
parser.add_argument('--column_idx',type=int,default=1,
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


# step-2: load in model
channel_size = [args.nhid]*(args.levels-1) + [args.emsize] # 这里可以自定义,建议调参看看
# n_cates = len(loader.dictionary.vals_set)  # emsize 为n_cates就是稀疏onehot向量，还可以更小就更稠密
n_cates = 100  #n_cates至少是args.emsize这么大
model = TrafficTCN(args.emsize,n_cates,channel_size,
                   kernel_size=args.ksize,dropout=args.dropout,
                   emb_dropout=args.emb_dropout,tied_weights=args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim,args.optim)(model.parameters(),lr=args.lr) # 反射调用
vis = visdom.Visdom(env="TrafficLine")

#step-3: train
def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    # vis.line(Y=train_data[0],X=torch.arange(0,720))
    for batch_idx,start_idx in enumerate(range(0,train_data.size(1),args.window_len)):
        """use sliding window to get per batch data and feed in net"""
        if start_idx+args.seq_len >= train_data.size(1):  # target的offset为1 所以取得等号不能执行之下逻辑
            continue # 边界条件再斟酌一下
        """data:n*seq_len    target:n*seq_len  """
        data,target = get_batch(train_data,start_idx,args)

        optimizer.zero_grad()

        """output:n_batch*seqlen*emb_size"""
        output = model(data)

        final_output = output.contiguous().view(-1,n_cates)
        final_target = target.contiguous().view(-1).to(torch.int64)

        loss = criterion(final_output,final_target)

        loss.backward()

        # writer.add_scalar("train_loss",loss.detach().numpy(),batch_idx)

        if args.clip >0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)

        optimizer.step()

        total_loss += loss.data

        """batch_idx 0到44  45=720/16"""
        if batch_idx%args.log_interval == 0 and batch_idx>0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| {:3d}/{:3d} batches | lr {:02.5f} | ms/batch {:5.5f} | loss {:5.2f} | ppl {:8.2f} |'.format(
                batch_idx,train_data.size(1)//args.window_len,lr,
                elapsed*1000/args.log_interval,cur_loss,math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

@torch.no_grad()
def evaluate(data_source,type):
    model.eval()
    total_loss = 0
    total_data_len = 0
    epoch_final_output = torch.FloatTensor([])
    epoch_final_target = torch.LongTensor([])
    for batch_idx, start_idx in enumerate(range(0, data_source.size(1), args.window_len)):
        """use sliding window to get per batch data and feed in net"""
        if start_idx + args.seq_len >= train_data.size(1):  # target的offset为1 所以取得等号不能执行之下逻辑
            continue  # 边界条件再斟酌一下
        """data:n*seq_len    target:n  """
        data, target = get_batch(data_source, start_idx, args)

        """output:n_batch*seqlen*n_cates"""
        output = model(data)

        final_output = output.contiguous().view(-1, n_cates) # n_cates
        final_target = target.contiguous().view(-1)
        loss = criterion(final_output, final_target)


        epoch_final_output = torch.cat([epoch_final_output,final_output[:args.window_len]],dim=0)
        epoch_final_target = torch.cat([epoch_final_target, final_target[:args.window_len]], dim=0)


        # writer.add_scalar(type+"_loss",loss.detach().numpy(),batch_idx)
        """就是加权算loss"""
        total_loss += args.window_len*loss.data
        total_data_len += args.window_len


    predict_values = torch.max(epoch_final_output, 1)[1]
    target_values = epoch_final_target
    val_len = target_values.size(0)
    # vis.line(Y=predict_values)
    # vis.line(Y=target_values)
    vis.line(Y=torch.stack([predict_values,target_values],dim=1),X=torch.arange(val_len),opts=dict(
        legend=["预测值","真实值"],xtrickstep=1,ytrickstep=1
    ))
    return total_loss / total_data_len

if __name__ == '__main__':
    best_vloss = 1e8
    lr = args.lr
    try:
        if IS_FIRST_RUN:
            all_val_loss = []
            # writer = SummaryWriter()
            for epoch in range(1,args.epochs+1):
                epoch_s_time =time.time()

                train()
                val_loss = evaluate(valid_data,"valid")
                test_loss = evaluate(test_data,"test")

                print("-"*89)
                print("| epoch {:3d} | time {:5.2f} | valid loss {:5.2f} | valid ppl {:8.2f}".format(
                    epoch,(time.time()-epoch_s_time),val_loss,math.exp(val_loss)
                ))
                print("| epoch {:3d} | time {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}".format(
                    epoch, (time.time() - epoch_s_time), test_loss, math.exp(test_loss)
                ))
                print("-"*89)

                if val_loss < best_vloss:
                    with open("checkpoint/model.pt","wb") as f: # 每次运行记得改这里
                        print("save model at model.pt")
                        torch.save(model,f)
                    best_vloss = val_loss


                if epoch>5 and val_loss> max(all_val_loss[-5:]):
                    lr = lr/2
                    for param in optimizer.param_groups:
                        param["lr"] = lr

                all_val_loss.append(val_loss)
        else:
            """get the best model on test data to reproduce the best result"""
            with open("checkpoint/model.pt", "rb") as f: #每次调试记得改这里
                model = torch.load(f)

            test_loss = evaluate(test_data, "test")
            print("=" * 89)
            print("End of train| test loss {:5.2f} | test ppl {:8.2f}".format(
                test_loss, math.exp(test_loss)
            ))
            print("=" * 89)
    except KeyboardInterrupt:
        print("-"*89)
        print("Existing later...")



























