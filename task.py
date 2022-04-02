'''
Author: Tammie li
Description: Define three-stage tasks
FilePath: \DRL\task.py
'''

import torch
import numpy as np
from tqdm import trange



def train_task(model, optimizer, data_loader, device, epochs, train_num, subject_id):
    print("===============Stage 1: representation learning =============")
    for epoch in trange(epochs):
        model.train()
        running_loss = 0.0
        correct_num = 0
        for index, data in enumerate(data_loader):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x, y = x.to(device), y.to(device)           
            y_pred = model(x, mode=0)
            loss = model.cal_loss(y_pred, y)
            correct_num += len([i for i in range(len(y)) if ((y[i] + y_pred[i]) < 1 and (y[i] + y_pred[i]) > 0)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        _loss = running_loss / train_num
        acc = correct_num / train_num * 100
        print(f'Train loss: {_loss+0.1:.4f}\tTrain acc: {acc:.2f}%')
        # valid_loss, valid_acc = valid_task(model, valid_loader, device, valid_num, subject_id)
    path = f'CheckPoint/{model.__class__.__name__}_{subject_id:>02d}_stage_1.pth'
    torch.save(model.state_dict(), path)

    print('Stage 1: <representation learning> Finish Training!')



def downstream_task(model, optimizer, criterion, data_loader, device, epochs, downstream_num, subject_id):
    print("===============Stage 2: classifier learning=============")
    model.load_state_dict(torch.load(f'CheckPoint/{model.__class__.__name__}_{subject_id:>02d}_stage_1.pth'))
    for epoch in trange(epochs):
        model.train()
        running_loss = 0.0
        correct_num = 0
        batch_size = None
        for index, data in enumerate(data_loader):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x, y = x.to(device), y.to(device)
            y = torch.tensor(y).to(torch.long)          
            y_pred = model(x, mode=1)
            loss = criterion(y_pred, y)
            _, pred = torch.max(y_pred, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += float(loss.item())
        batch_num = downstream_num // batch_size
        _loss = running_loss / (batch_num + 1)
        acc = correct_num / downstream_num * 100
        print(f'Downstream task loss: {_loss:.4f}\tDownstream task acc: {acc:.2f}%') 

    path = f'CheckPoint/{model.__class__.__name__}_{subject_id:>02d}_stage_2.pth'
    torch.save(model.state_dict(), path)
    print('Stage 2: <classifier learning> Finish Training!')



def test_task(model, criterion, DEVICE, data_loader, test_num, subject_id):
    print("===============Stage 3: Test model=============")
    model.load_state_dict(torch.load(f'CheckPoint/{model.__class__.__name__}_{subject_id:>02d}_stage_2.pth'))
    correct_num = 0
    model.eval()
    batch_size = None
    preds = []
    ys = []
    for index, data in enumerate(data_loader):
        x, y = data
        batch_size = x.shape[0] if index == 0 else batch_size
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        y_pred = model(x, mode=2)
        _, pred = torch.max(y_pred, 1)

        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
        preds.extend(pred.cpu().numpy().tolist())
        ys.extend(y.cpu().tolist())
        
    acc = correct_num / test_num * 100
    np.save(f'PredictionResult/{subject_id:>02d}_y', ys)
    np.save(f'PredictionResult/{subject_id:>02d}_preds', preds)

    print('Stage 3: <Test model> Finish Test!')
