import json
import torch
import torch.nn as nn
import torch.optim as optim

from Data.load_data import *
from task import *
from model import *

from Utils.evaluate import *
from Utils.record import *


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)


if __name__ == '__main__':

    # parameter setting
    with open('config.json', 'r') as f:
        config_dict = json.load(f)

    Dataset_info = config_dict['dataset']
    model_info = config_dict['model_para']
    train_info = config_dict['train_para']

    model_info['model'] = DRL(n_class=2, channels=Dataset_info['channel'], f1=model_info['F1'],
                                f2=model_info['F2'], d=model_info['D'], drop_out=model_info['droup_out'],
                                kernel_length=model_info['kernel_size']).to(DEVICE)

    _subject_id = [i for i in range(1, Dataset_info['subject_num']+1)]
    _mean = [i for i in range(1, 6)]

    # Select optimizer
    optimizer = optim.Adam(model_info['model'].parameters())

    # The loss function of classifier stage 
    criterion = nn.CrossEntropyLoss()

    for subject_id in _subject_id:
        write_table_head(subject_id, model_info, Dataset_info["name"])
        target, non_target, x_test, y_test = load_dataset(subject_id, Dataset_info['name']) 
        data_info = generate_data_info(target, non_target, x_test, y_test, subject_id, train_info['batchsize_stage_1'], 
                                        train_info['batchsize_stage_2'], Dataset_info['pairNum'])
        for mean in _mean:
            train_task(model_info['model'], optimizer, data_info['train_loader'],
                        DEVICE, train_info['epoch_stage_1'], data_info['train_num'], subject_id)
            downstream_task(model_info['model'], optimizer, criterion, data_info['downstream_loader'],
                            DEVICE, train_info['epoch_stage_2'], data_info['downstream_num'], subject_id)
            test_task(model_info['model'], criterion, DEVICE, data_info['test_loader'], data_info['test_num'], subject_id)
            uar, war = calculate_UAR_for_single_subject(subject_id), calculate_WAR_for_single_subject(subject_id)
            write_result(subject_id, model_info, Dataset_info['name'], mean, uar, war)
        


