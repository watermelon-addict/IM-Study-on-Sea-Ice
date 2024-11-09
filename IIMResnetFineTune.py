# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:17:29 2024

ResNet50 model setup and fine tune training, based on PyTorch machine learning framework.

Functions in this script are called by IIM_Resnet_Results.py to solve the Inverse Ising Problem.  

Functions:
create_Resnet_Model():      Set up the ResNet50 network with final FC layer of 5 neurons
train_Resnet():             module to train ResNet50 network
IIM_Resnet_Run():           Load data and train
    
@author: Ellen Wang
"""

import numpy as np
import json
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
if( "D:\\Users\\junwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\junwang\\source\\py\\ICEIsing" )

from ReadSeaIce import read_SI
from IIMSimul import IIM_period_test
from IIMConstants import v1, v2, NX, NY, metrosteps, NumParam
import IceIsingCont as IIMC # Ice Ising Model Continuous
from IIMCNNModel import LoadData, YMD_start, YMD_end

# formatting templates
ymd = "{:4d}_{:02d}_{:02d}"
param_fmt = "{:04.2f}_{:04.2f}_{:04.2f}_{:05.2f}_{:05.2f}"

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.backends import cudnn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True

from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def train_Resnet(model, criterion, optimizer, scheduler, 
                 dataloaders, dataset_sizes, device, num_epochs=10):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    # with TemporaryDirectory() as tempdir:
    tempdir = ".\\temp_res"
    best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

    best_loss = 9.9e9

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(f"inputs shape {inputs.shape}, target shape {labels.shape}")
                    outputs = model(inputs)
                    # print(f"inputs shape {inputs.shape}, target shape {labels.shape}")
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # save model with lowest validation loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def create_Resnet_Model( NL=50, wgts='IMAGENET1K_V1' ):
    # fully trained model
    model_ft = models.resnet50(weights=wgts)
    # model_ft = models.resnet50(weights='IMAGENET1K_V1')
    # replace input with image of 2 channels
    model_ft.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    # set last fully connected layer to the desired number of output classes                           
    num_ftrs = model_ft.fc.in_features
    num_out_ftrs = model_ft.fc.out_features
    
    # if keeping the last fully connected layer, just change the number of outputs
    # model_ft.fc = nn.Linear(num_ftrs, NumParam )

    # model_ft = model_ft.to(device)
    # params = model_ft.parameters()    
    # print(model_ft)
    
    # add another linear module
    lin = model_ft.fc
    new_lin = nn.Sequential(
        lin,
        nn.ReLU(),
        nn.Linear(num_out_ftrs, NumParam)
        )
    model_ft.fc = new_lin
    
    return model_ft


def IIM_Resnet_Run( year=2023 ):
    data = LoadData( year )
        
    # set up training data
    # each X contains 2 channels which are the images of the initial and end state
    # Y are the Ising parameters
    X = []
    Y = []
    for dd, vv in data.items():
        y1,m1,d1 = dd.split('_')
        y1 = int(y1)
        m1 = int(m1)
        d1 = int(d1)
        
        s1 = read_SI( y1, m1, d1) # initil state
        for param_str, s2 in vv.items():        
            params = param_str.split( '_' )
            params = [ float(x) for x in params ]
            if( True ):
                s2 = np.array(s2)

                # Tensorflow data shape (N, NX, NY, C), channel at the end
                # Pytorch data shape(N, C, NX, NY), channel before x and y
                X.append(np.stack((s1,s2), axis=0)) # pytorch has channel at the beginning
                Y.append( np.array(params) )        
    
                # show some training samples, every one out of 10,000 samples
                if( np.random.random() > 0.9999 ):
                    fig, ax = plt.subplots(1,2)
                    ax[0].imshow( s1, cmap="Blues", vmin=v1, vmax=v2 )
                    ax[1].imshow( s2, cmap="Blues", vmin=v1, vmax=v2 )
                    ax[0].set_title('(a)', y = -0.18)
                    ax[1].set_title('(b)', y = -0.18)
                    fig.suptitle( dd + "   " + param_str )
                    plt.show()
    del data # free memory
    # X=X[0:10000]
    # Y=Y[0:10000]
    print(len(X))
        
    # check training data IM parameters
    # for k, d in data["2023_10_01"].items():
    #     print(k)
    # whether or not normalizing the y_data
    normalizing = False # set to false if use original trainY
    # means = np.mean(Y, axis=0)
    # stds = np.std(Y, axis=0)
    # print(means)
    # print(stds)
    means = np.array([2.49967741, -3.92932612, -5.29179677, -0.05106308, 10.000485])
    stds = np.array([0.14406463, 8.92044723, 2.75589219, 5.76719388, 0.57913891])
    
    if(normalizing):
        Y = ( Y - means ) / stds
    # print(Y)
    
    # split train and test data. We don't need a lot of testing data here
    split = train_test_split(X, Y, test_size=0.01, random_state=42)
    (trainX, testX, trainY, testY) = split
    
    trainX = torch.from_numpy(np.array(trainX, dtype="float32"))
    trainY = torch.from_numpy(np.array(trainY, dtype="float32"))
    testX = torch.from_numpy(np.array(testX, dtype="float32"))
    testY = torch.from_numpy(np.array(testY, dtype="float32"))
    
    trainsz = len(trainX)
    testsz = len(testX)
    
    # free memory
    del X
    del Y
    del split
    
    # print some stats for debugging purpose only 
    print( "size of training data: ", trainsz )
    print( "size of testing data: ", testsz )    
    print("[INFO] training model...")
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
    model_ft = create_Resnet_Model()
    model_ft.to(device)
    print(model_ft)

    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
    batch_size = 128
    epochs = 30
    
    trainds = TensorDataset(trainX, trainY)
    testds = TensorDataset(testX, testY)
    train_dl = DataLoader(trainds, batch_size=batch_size, shuffle=True )
    test_dl = DataLoader(testds, batch_size=batch_size, shuffle=True )    
    dataloaders = {"train": train_dl, "val": test_dl}
    dataset_sizes = {"train": len(trainds), "val": len(testds) }
        
    num_params = sum([p.numel() for p in model_ft.parameters()])
    trainable_params = sum([p.numel() for p in model_ft.parameters() if p.requires_grad])
    print(f"{num_params = :,} | {trainable_params = :,}")
    
    # check model input/output - if the model runs on correct data shapes
    # out = model_ft(testX[0:1].to(device)).cpu()
    # print(out)
    # print(testY[0:1])
    # diff = criterion(out, testY[0:1])
    # print(diff)
    
    model_ft = train_Resnet(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 
                            dataloaders, dataset_sizes, device, num_epochs=epochs)

    # done with ConvNet as fully trainged model
    torch.save(model_ft.state_dict(), ".\\ResNet_models\\resnet50_ft_IIM_2022_21.pt")
        
    # load saved model
    # model_ft.load_state_dict(torch.load(".\\ResNet_models\\resnet50_ft_IIM_Norm_3.pt"))
    # num_params = sum([p.numel() for p in model_ft.parameters()])
    # trainable_params = sum([p.numel() for p in model_ft.parameters() if p.requires_grad])
    # print(f"{num_params = :,} | {trainable_params = :,}")
    
    # check model performance
    check = True
    # year = 2023
    ymdf = "{:4d}_{:02d}_{:02d} to {:4d}_{:02d}_{:02d}"    
    year_params = []
    model_ft.eval()
    if( check ):
        # the below code check all periods of 2023, but you can check choose to check a single period of course
        ymd1 = YMD_start(year)
        ymd2 = YMD_end(year)
            
        sz = len(ymd1)
        for i in range(sz):
            y1, m1, d1 = ymd1[i]
            y2, m2, d2 = ymd2[i]
            s1 = read_SI( y1, m1, d1)
            s2 = read_SI( y2, m2, d2)
            print( "start", y1, m1, d1, "    end", y2, m2, d2 )    
    
            s = np.stack((s1,s2), axis=0)
            s = np.array(s, dtype="float32").reshape((1, 2, NX, NY))        
            s = torch.from_numpy(s).to(device)
            
            param_ch = model_ft(s)                
            param_np = param_ch.reshape( NumParam ).cpu().detach().numpy()

            if(normalizing):
                param_np = param_np * stds + means
                
            year_params.append(param_np)
            print(param_np)
            ymd_str = ymdf.format(y1,m1,d1,y2,m2,d2)
            param_str = param_fmt.format(param_np[0],param_np[1],param_np[2],param_np[3],param_np[4])
            
            # IIM_period_test displays 3 images for each period so we can visualize how the model works for each period
            # 3 images: the initial lattice, the target end lattice and the CNN predicted target end lattice
            d, s1_outs = IIM_period_test( param_np, args = [s1, s2, NX, NY], plt_title=ymd_str+"\n"+param_str)
    
    year_params = np.array(year_params)
    print(year_params)

# IIM_Resnet_Run( 2022 )