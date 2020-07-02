import codecs
import os
import argparse
import time
import csv
import pickle
import operator
from datetime import datetime
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = 'user_log_format1.csv'

print(dataset)

if dataset == 'user_log_format1.csv':

    print("begin")
    # df = pd.read_csv(dataset, nrows=1000)
    df = pd.read_csv(dataset)
    # mylist = []
    # count = 0
    # for chunk in pd.read_csv(dataset, chunksize=100000000):
    #     count = count + 100000000
    #     print(count)
    #     mylist.append(chunk)
    #     df = pd.concat(mylist, axis=0)
    # del mylist
    print("end")
    # df = pd.read_csv(dataset)
    # print(df['user_id'])

    print("begin time_stamp")
    df = df[(df['action_type'] == 2) & (df['time_stamp'] <= 1112)]

    # print(df)

    print("begin co_user")
    group = df.groupby(['item_id'])
    df = group.filter(lambda x : x['user_id'].count() > 20)

    print("begin sess_len")
    group = df.groupby(['user_id', 'time_stamp'])
    df = group.filter(lambda x : x['item_id'].count() > 1)

    print("begin user_sess")
    group = df.groupby(['user_id'])
    # user_sess大于1即可，如果有拆分long和short，那么只用作训练数据，测试集就不要了
    df = group.filter(lambda x : len(x['time_stamp'].unique()) > 1)

    # print("begin user")
    # group = df.groupby(['user_id'])
    # print(len(group))
    #
    # print("begin item")
    # group = df.groupby(['item_id'])
    # print(len(group))

    print("begin to_csv")
    df.to_csv('result.csv',index=0)