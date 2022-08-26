#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import math
import random
import time
import csv
import pickle
import operator
import datetime
import os

import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/dressipi/sample')
parser.add_argument('--date', default="2021-04", type=str, help='start date of training dataset')
parser.add_argument('--cut', action='store_true', help='cut the test session')
opt = parser.parse_args()
print(opt)


dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'
elif opt.dataset == "dressipi":
    dataset = "dressipi/train_sessions_purchases.csv"


print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    elif opt.dataset == "dressipi":
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}      # 這個看起來好像只會存 session 的最後一個 date ??
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        # 當換新的 sessid 的時候，把 date 存入
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                # print(f"date: {date}")
            if opt.dataset == "dressipi":
                date = time.mktime(time.strptime(curdate[:19], "%Y-%m-%d %H:%M:%S"))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        elif opt.dataset == "dressipi":
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        elif opt.dataset == "dressipi":
            curdate = data['date']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    if opt.dataset == "dressipi":
        date = time.mktime(time.strptime(curdate[:19], "%Y-%m-%d %H:%M:%S"))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# the date of sessions
dates = list(sess_date.items())


##########################################################
# 依照時間切成 train 和 test
##########################################################
splitdate = "2021-05-01 00:00:00"
splitdate = time.mktime(time.strptime(splitdate, "%Y-%m-%d %H:%M:%S"))
print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    # item_ctr: 就是 main.py 裏面的 n_node, 代表不重複的 item 數量, 也就是 GNN 的 node 的數量
    # 43098 (diginetica), 37484 (yoochoose1_64 & yoochoose1_4)
    print(f"item_ctr (n_node): {item_ctr}")
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():  # obtian?? lol
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates, cut):
    """
    Args:
        cut: True or False
             True 的話會做 session 的切片，False 則不會做切片
    """
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    # train，一定要做切片
    if cut:
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
    # test，可選擇要不要切片
    else:
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            labs += [seq[-1]]
            # 在這裡加入隨機切 test session 的長度
            if opt.cut:
                cut_len = random.uniform(0.5, 1.0) * len(seq)
                cut_len = math.ceil(cut_len)
                # 讓他不會讀到 purchase
                if cut_len == len(seq):
                    cut_len = cut_len - 1
                out_seqs += [seq[:cut_len]]
            # 不要隨機切 test session, RecSys Challenge 2022 的其中一項規定
            elif not opt.cut:
                out_seqs += [seq[:-1]]
            out_dates += [date]
            ids += [id]

    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, cut=True)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, cut=opt.cut)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))

elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

elif opt.dataset == "dressipi":
    split_y_m = opt.date

    # 根據 test session 有沒有切分不同資料夾
    if opt.cut:
        save_dir = f"dressipi_d{split_y_m}_cut_pur"
    elif not opt.cut:
        save_dir = f"dressipi_d{split_y_m}_nocut_pur"

    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"save dir: {save_dir}")
    pickle.dump(tes, open(f"{save_dir}/test.txt", "wb"))

    # split64 = int(len(tr_seqs) / 64)
    # print(len(tr_seqs[-split64:]))

    # 在這裡切 train 的時間
    train_splitdate = f"{split_y_m}-01 00:00:00"
    train_splitdate = time.mktime(time.strptime(train_splitdate, "%Y-%m-%d %H:%M:%S"))
    split = [i for i in range(len(tr_dates)) if tr_dates[i] > train_splitdate]  # 回傳所有符合條件的 index
    split = split[0]  # 第一個就是最小的那個

    tra = (tr_seqs[split:], tr_labs[split:])
    seq = tra_seqs[tr_ids[split]:]

    # tra64 = (tr_seqs, tr_labs)
    # seq64 = tra_seqs

    pickle.dump(tra, open(f"{save_dir}/train.txt", "wb"))
    pickle.dump(seq, open(f"{save_dir}/all_train_seq.txt", "wb"))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
