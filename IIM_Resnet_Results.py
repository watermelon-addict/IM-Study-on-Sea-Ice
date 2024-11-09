# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:56:40 2024

Functions for collect and save Ising ResNet50 results

collect_ResNet_res():      Collect ResNet50 results for multiple periods of a year, save the results

These functions are called by Gen_Figures.py

@author: Ellen Wang
"""
import keras
import IIMSimul
import numpy as np
import json
import datetime
from datetime import date
import torch

import sys
if( "D:\\Users\\junwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\junwang\\source\\py\\ICEIsing" )
from ReadSeaIce import read_SI
from IIMConstants import v1, v2, NX, NY, metrosteps, NumParam
from IIMCNNModel import YMD_start, YMD_end
from IIM_CNN_Results import save_CNN_res
from IIMResnetFineTune import create_Resnet_Model
import IIMSimul
import datetime

# Saved ResNet models
ResnetModelMap = { 2023: 2, 2012: 11, 2022: 12 }

# Collect mulitple periods of a full year, save the results
def collect_Resnet_res( year=2023 ):
    msf = ".\\ResNet_models\\resnet50_ft_IIM_{:d}.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_Resnet_Model()
    
    num_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"{num_params = :,} | {trainable_params = :,}")
    # print(model)

    # means and stds used for the case of normalizing the outputs
    # production set normalizing to False which gives better results
    normalizing = False
    means = np.array([2.49967741, -3.92932612, -5.29179677, -0.05106308, 10.000485])
    stds = np.array([0.14406463, 8.92044723, 2.75589219, 5.76719388, 0.57913891])

    # A single model for the whole year
    ymd1 = YMD_start( year )
    ymd2 = YMD_end( year )
    sz = len(ymd1)

    for i in range(sz):
        y1,m1,d1 = ymd1[i]
        y2,m2,d2 = ymd2[i]

        mn = ResnetModelMap[ year ] 
        ms = msf.format(mn)
        model.load_state_dict(torch.load(ms))
        print( f"loaded {ms}" )
        model.to(device)

        s1 = read_SI( y1, m1, d1)
        s2 = read_SI( y2, m2, d2)
        s = np.stack((s1,s2), axis=0)
        s = np.array(s).reshape((1, 2, NX, NY))        
        s = torch.from_numpy( np.array(s, dtype="float32") ).to(device)
        model.eval()
        param_ch = model(s)
        param_ch = param_ch.reshape( NumParam ).cpu().detach().numpy()

        if(normalizing):
            param_ch = param_ch * stds + means
        
        save_CNN_res(y1,m1,d1, y2,m2,d2, param_ch, Resnet=True)
        
        # testing only
        # ymdf = "{:4d}_{:02d}_{:02d} to {:4d}_{:02d}_{:02d} "
        # ymd = "{:4d}_{:02d}_{:02d}"
        # param_fmt = "{:04.2f}_{:04.2f}_{:04.2f}_{:05.2f}_{:05.2f}"
        # param_ch = np.array([2.725, -12.69, -6.09, -8.42, 9.32])
        # ymd_str = ymdf.format(y1,m1,d1,y2,m2,d2)
        # param_str = param_fmt.format(param_ch[0],param_ch[1],param_ch[2],param_ch[3],param_ch[4])
        # d, s1_outs = IIMSimul.IIM_period_test( param_ch, args = [s1, s2, NX, NY], plt_title=ymd_str+"\n"+param_str)

# collect_Resnet_res( year=2022 )
