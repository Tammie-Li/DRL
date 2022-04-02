'''
Author: Tammie li
Description: data operate
FilePath: \DRL\Data\load_data.py
'''
import os
import numpy as np
import random

from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn import preprocessing
from Data.make_dataset import *


def scale_data(data):
    scaler = preprocessing.StandardScaler()
    for i in range(data.shape[0]):
        data[i, :, :] = scaler.fit_transform(data[i, :, :])
    return data

def load_dataset(subject_id, dataset_type):
    # 加载数据
    print(f'Current directory: s{subject_id}')

    target_data = np.load(os.path.join(os.getcwd(), 'Data', dataset_type, f'S{subject_id:>02d}', 'target.npy'))
    non_target_data = np.load(os.path.join(os.getcwd(), 'Data', dataset_type, f'S{subject_id:>02d}', 'nontarget.npy'))
    x_test = np.load(os.path.join(os.getcwd(), 'Data', dataset_type, f'S{subject_id:>02d}', 'x_test.npy'))
    y_test = np.load(os.path.join(os.getcwd(),'Data', dataset_type, f'S{subject_id:>02d}', 'y_test.npy'))

    target_data = scale_data(target_data)
    non_target_data = scale_data(non_target_data)
    x_test = scale_data(x_test)
    
    print("Shape of target dataset", target_data.shape)
    print("Shape of non-target dataset", non_target_data.shape)
    print("Shape of test dataset", x_test.shape)

    return target_data, non_target_data, x_test, y_test

# Sample pair construction
def generate_pair(target_data, non_target_data, num):
    tmp_data = []
    pair_data = []

    # Paired target + target as positive pairs
    for i in range(int(num/4)):
        perm = np.random.permutation(target_data.shape[0])
        rd_1, rd_2 = random.randint(0, target_data.shape[0]-1), random.randint(0, target_data.shape[0]-1)
        tmp_data = []
        tmp_data.append(target_data[rd_1, ...])
        tmp_data.append(target_data[rd_2, ...])
        pair_data.append(tmp_data)

    # Paired nontarget + nontarget as positive pairs
    for i in range(int(num/4)):
        rd_1, rd_2 = random.randint(0, non_target_data.shape[0]-1), random.randint(0, non_target_data.shape[0]-1)
        tmp_data = []
        tmp_data.append(non_target_data[rd_1, ...])
        tmp_data.append(non_target_data[rd_2, ...])
        pair_data.append(tmp_data)

    
    # Paired target + nontarget as negative pairs
    for i in range(int(num/2)):
        rd_1, rd_2 = random.randint(0, target_data.shape[0]-1), random.randint(0, non_target_data.shape[0]-1)
        tmp_data = []
        tmp_data.append(target_data[rd_1, ...])
        tmp_data.append(non_target_data[rd_2, ...])
        pair_data.append(tmp_data)
    
    pair_label = [0 for i in range(int(num/2))] + [1 for i in range(int(num/2))]

    pair_data, pair_label = np.array(pair_data), np.array(pair_label)

    return pair_data, pair_label


def generate_data_info(target_data, non_target_data, x_test, y_test, subject_id, batch_size_1, batch_size_2, num):

    # Data for stage 1: representation learning 
    x_train, y_train = generate_pair(target_data, non_target_data, num)
    print("Shape of paired full dataset for train", x_train.shape)

    # Data for stage 2: classifier learning
    perm = np.random.permutation(non_target_data.shape[0])
    non_target_num = int(target_data.shape[0])
    non_target_data_downstream = non_target_data[perm[:non_target_num]]

    x_downstream = np.concatenate((non_target_data_downstream, target_data))
    y_downstream = [0 for i in range(non_target_data_downstream.shape[0])] + [1 for i in range(target_data.shape[0])]
    y_downstream = np.array(y_downstream)

    # Make Dataset
    train_data = TrainDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size_1, shuffle=True)

    downstream_data = DownstreamDataset(x_downstream, y_downstream)
    downstream_loader = DataLoader(downstream_data, batch_size=batch_size_2, shuffle=True)

    test_data = TestDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size_2, shuffle=False)

    # Construct data info
    data_info = defaultdict()
    data_info['subject_id'] = subject_id
    data_info['times'] = x_train.shape[-1]
    data_info['channels'] = x_train.shape[-2]

    data_info['train_num'] = x_train.shape[0]
    data_info['downstream_num'] = x_downstream.shape[0]
    data_info['test_num'] = x_test.shape[0]

    data_info['train_loader'] = train_loader
    data_info['downstream_loader'] = downstream_loader
    data_info['test_loader'] = test_loader

    return data_info