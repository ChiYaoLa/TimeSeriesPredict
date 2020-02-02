# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 14:52
# @Author  : XuLiang1995
import os
import torch
import pickle


def data_generator(args):
    loader_path = os.path.join(args.data,"loader.pt")
    if  os.path.exists(loader_path):
        TensorData = pickle.load( open(loader_path,"rb"))
    else:
        TensorData = TensorFactory(args.data)
        pickle.dump(TensorData,open(loader_path,"wb"))
    return TensorData

class Dictionary:
    def __init__(self):
        self.vals_set = [] # contain of speed_value/volumn_value
        self.cate2val = {} #from cate_idx to speed_value/volumn_value
        self.val2cate = {}

    def add_val(self,val):
        if val not in self.vals_set:
            self.vals_set.append(val)
            self.cate2val[len(self.vals_set)-1] = val
            self.val2cate[val] = len(self.vals_set)-1
        return val


    def __len__(self):
        return len(self.vals_set)


class TensorFactory:
    def __init__(self,path):
        self.dictionary = Dictionary()
        self.train = self.to_tensor(os.path.join(path,"train.csv"))
        self.test = self.to_tensor(os.path.join(path,"test.csv"))
        self.valid = self.to_tensor(os.path.join(path,"valid.csv"))

    def to_tensor(self,path):
        print(path)
        assert os.path.exists(path)
        data = []
        with open(path,"r") as f:
            for line in f:
                try:
                    nums = list(map(int,line.split(",")[1:]))# 暂时先不考虑第一列-时间戳
                except ValueError:
                    print("Error Line:")
                    print(line)
                """add-in  """
                for i in range(len(nums)):
                    self.dictionary.add_val(nums[i])
                    # nums[i] = self.dictionary.val2cate[nums[i]]
                data.append(nums)

        return torch.LongTensor(data)



def batchfy(data,batch_size,args):
    """
    :param data: 原始csv直转的二维tensor, 56*720 56=14*4,或者固定断面 14*720  14天的
    :return: n*720
    """
    fix_column_idx = args.column_idx-1 # 先不考虑第一列时间戳
    data = torch.narrow(data,dim=1,start=fix_column_idx,length=1) # 取得某列所有值
    data = data.view(-1,720) # n*720
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source,start_idx,args):
    """
    :param source: n*720
    :return: data:n*seq_len ,target:n*seq_len
    """
    data = source[:,start_idx : start_idx+args.seq_len]
    target = source[:,start_idx+1 : start_idx + args.seq_len + 1]
    return  data,target


def get_input_curve(writer,data_tensor,type):
    name = type + "_volumn"
    # for i in range(data_tensor.size(1)):
    for i in range(3):
        print({
            str(k):v  for k, v in enumerate(data_tensor[:, i].detach().view(-1).numpy())
        })
        writer.add_scalars(name,{
            str(k):v  for k, v in enumerate(data_tensor[:, i].detach().view(-1).numpy())
        },i)

    return None

def get_output_curve(writer,final_output,type):
    """output: N * C tensor，you must extract cate_idx from sequence of (0,1) values ：marked by xuliang"""
    name = type + "_volumn"
    cate_idxes = torch.max(final_output,1)[1].numpy()
    steps = final_output.size(0)
    for i in range(steps):
        writer.add_scalar(name,cate_idxes[i],i)
    return None

# data = torch.rand((2,3))
# print(data)
# print(get_output_curve("",data,""))



















