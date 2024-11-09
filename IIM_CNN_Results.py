# -*- coding: utf-8 -*-
"""
Created on Sat Dec 9 12:04:28 2023

Functions for collect and save Ising CNN results

Save_CNN_Res():		Save CNN Ising parameters results and the lattice for a single period.
                    This is used to save ResNet50 and ViT results too.
                    
collect_CNN_res():	Collect multiple periods of a year, save CNN results

These functions are called by Gen_Figures.py

@author: Ellen Wang
"""

import keras
import IIMSimul
import numpy as np
import json
import datetime
from datetime import date

import sys
if( "D:\\Users\\junwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\junwang\\source\\py\\ICEIsing" )
from ReadSeaIce import read_SI
from IIMConstants import v1, v2, NX, NY, metrosteps, NumParam
from IIMCNNModel import YMD_start, YMD_end

# Saved CNN models
CNNModelMap = { 2022: 25, 2023: 32, 2012: 39 }

# save the CNN predicted Ising parameters and the lattice result in ..\\IIM_CNN_Outputs\\result_{:s}_{:s}_rundate{:s}.json
def save_CNN_res(y1,m1,d1, y2,m2,d2, params, Resnet=False, ViT=False):
    s1 = read_SI( y1, m1, d1)
    s2 = read_SI( y2, m2, d2)
    d, s1_outs = IIMSimul.IIM_period_test(params, args = [s1, s2, NX, NY])

    ymd = "{:4d}{:02d}{:02d}"
    ymd1 = ymd.format(y1, m1, d1)
    ymd2 = ymd.format(y2, m2, d2)
    tod = str(date.today())
    if(ViT):
        of = '..\\IIM_ViT_Outputs\\result_prod_{:s}_{:s}_rundate{:s}.json'  
    elif(Resnet):
        of = '..\\IIM_Resnet_Outputs\\result_prod_{:s}_{:s}_rundate{:s}.json'  
    else:
        of = '..\\IIM_CNN_Outputs\\result_{:s}_{:s}_rundate{:s}.json'
    of = of.format( ymd1, ymd2, tod )    
    
    l = s1_outs[0].tolist()
    res = {"start": (y1,m1,d1), 
           "end": (y2,m2,d2), 
           "retx": params.tolist(),
           "array": l,
           "metrosteps": metrosteps,    # dummy returns
           "retfun": 0,                 # dummy returns
           "success": True              # dummy returns
           }
    with open( of, "w") as outfile:
        json.dump( res, outfile, indent = 4)    

    print( f"Saved {of}")

# Collect mulitple periods of a full year, save the results
def collect_CNN_res( year=2022 ):
    msf = "IIM_CNN_model{:d}"

    mn = CNNModelMap[ year ]
    ymd1 = YMD_start( year )
    ymd2 = YMD_end( year )
    sz = len(ymd1)
    for i in range(sz):
        y1,m1,d1 = ymd1[i]
        y2,m2,d2 = ymd2[i]

        ms = msf.format(mn)
        model = keras.models.load_model(ms)

        s1 = read_SI( y1, m1, d1)
        s2 = read_SI( y2, m2, d2)
        s = np.stack((s1,s2), axis=-1)
        s = np.array(s).reshape((1, NX, NY, 2))        

        param_ch = model.predict( s )
        param_ch = param_ch.reshape( NumParam )
                
        save_CNN_res(y1,m1,d1, y2,m2,d2, param_ch)
        
