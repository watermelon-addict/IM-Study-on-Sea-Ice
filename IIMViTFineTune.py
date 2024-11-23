# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:17:29 2024

ViT model setup and fine tune training, based on Huggingface Transformer package.

Functions in this script are called by IIM_ViT_Results.py to solve the Inverse Ising Problem.  

Functions:
create_ViT_Model():      Set up the fine-tuned ViT network with final FC layer of 5 neurons
IIM_ViT_Run():           Load data and train
collate_fn():            Auxilliary function to set up data in the format of huggingface ViT
RegressionTrainer():     Class for regression MSE error
compute_metrics_for_regression():   Metrics during training process
np_one_hot():            Simple one-hot encoding

@author: Ellen Wang
"""

import numpy as np
import json
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
if( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" )

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
from torch.nn import Conv2d
from torch.nn.functional import one_hot

import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader, TensorDataset

import os
from PIL import Image
from tempfile import TemporaryDirectory
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# transformers
import transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from transformers import logging

# display where the checkpoints are saved
logging.set_verbosity_info()
logging.enable_progress_bar()

cudnn.benchmark = True

img_size = 60
num_Ch = 2
patch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
device2 = torch.device("cuda")

def collate_fn(examples):
    pixels = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    return {"pixel_values": pixels, "labels": labels}


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ls = inputs.pop("labels")
        outputs = model(**inputs)
        # logits = outputs[0][:, 0]
        logits = outputs[0]
        loss = torch.nn.functional.mse_loss(logits, ls)
        return (loss, outputs) if return_outputs else loss

def np_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def compute_metrics_for_regression(eval_pred):
    logits, ls = eval_pred
    
    mse = mean_squared_error(ls, logits)
    mae = mean_absolute_error(ls, logits)
    r2 = r2_score(ls, logits)
    
    # Compute accuracy 
    accuracy = 1.0 / mse
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


def create_ViT_Model( model_name='google/vit-base-patch16-224' ):
    # tried different models
    # model_name = "google/vit-base-patch16-224"
    # model_name = "facebook/deit-base-patch16-224"
    # model_name = "facebook/deit-tiny-patch16-224"
    # model_name = "microsoft/beit-base-patch16-224-pt22k"
    
    model = ViTForImageClassification.from_pretrained(model_name, 
                                                       num_labels=NumParam, num_channels=2, 
                                                      # num_labels=lin_out_num, num_channels=2, 
                                                      image_size=img_size, patch_size=patch_size,
                                                      ignore_mismatched_sizes=True)
    
    return model



def IIM_ViT_Run():
    data = LoadData( 2023 )
        
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
    X=X[30000:30100]
    Y=Y[30000:30100]
    print(len(X))

    ds = [ list(a) for a in zip(X,Y) ]
            
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
    split = train_test_split(ds, test_size=0.01, random_state=42)
    (tds1, tds2) = split
    
    
    pixels = torch.from_numpy(np.array([e[0] for e in tds1], dtype="float32")).to(device)
    labels = torch.from_numpy(np.array([e[1] for e in tds1], dtype="float32")).to(device)
    trainds = TensorDataset(pixels, labels)

    pixels = torch.from_numpy(np.array([e[0] for e in tds2], dtype="float32")).to(device)
    labels = torch.from_numpy(np.array([e[1] for e in tds2], dtype="float32")).to(device)
    testds = TensorDataset(pixels, labels)
    trainsz = len(trainds)
    testsz = len(testds)
    # print some stats for debugging purpose only 
    print( "size of training data: ", trainsz )
    print( "size of testing data: ", testsz )    
    
    
    # free memory
    del ds
    del X
    del Y
    del split
    del pixels
    del labels
    
    t=trainds[0][0]
    t = torch.unsqueeze(t, 0)
    print(t)
    t.shape
    t.to(device)
    
    l=trainds[0][1]
    print(l)

    model = create_ViT_Model( model_name='google/vit-base-patch16-224' )
    model.to(device)
    print(model)    
    num_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"{num_params = :,} | {trainable_params = :,}")
    
    out = model(t)
    print(out)

    args = TrainingArguments(
        "IIM_vit-reg",
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    
    trainer = RegressionTrainer(
        model,
        args, 
        train_dataset=trainds,
        eval_dataset=testds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_for_regression,
        tokenizer=None,
        callbacks = [transformers.ProgressCallback()], # show epoch progress
    )
    
    print("\n\n...Start training ViT model...")
    start = time.time()
    trainer.train()
    end = time.time()
    print("Training time is:%6.0f seconds" % (end-start) )
    
    outputs = trainer.predict(testds)
    print(outputs)
    print(outputs.metrics)
        
    # save model to local and load 
    print("\n\n...saving model...")
    trainer.save_model(".\\ViT_models\\google_vit-base-patch16-224_3")

    # check model performance
    check = True
    year = 2023
    ymdf = "{:4d}_{:02d}_{:02d} to {:4d}_{:02d}_{:02d}"    
    year_params = []
    model.eval()
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
            s = torch.from_numpy(s).to(device2)
            
            param_ch = model(s)  
            # print(param_ch[0])              
            param_np = param_ch[0].squeeze().cpu().detach().numpy()

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

# IIM_ViT_Run()