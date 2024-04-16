# -*- coding: utf-8 -*-
"""
Created on Sat Sep 9  14:10:50 2023

1. Generate random Ising parameters and run metropolis simulation, save the simulation results in 
..\\IIMParamGen\\IIMParamGen_{:s}.json. These results are used as inputs for CNN model training.
The code runs in parallel; and it takes many hours to run. Please be cautious before start running this

2. Train and save the IIM_CNN model 

@author: Ellen Wang
"""

import sys
if( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" )

from IIMCNNModel import CNNTrain, MultiPeriodGen

GenParam = False # set to True if generating CNN training data
TrainModel = False # set to True if train the CNN model

if ( GenParam and __name__ == '__main__'): # usually do not run
    MultiPeriodGen( 2022, NN=10000, savefile=True )
    MultiPeriodGen( 2023, NN=10000, savefile=True )
    MultiPeriodGen( 2012, NN=10000, savefile=True )

if( TrainModel ):
    CNNTrain( 2022, modelNum=101, check=False )
    CNNTrain( 2023, modelNum=102, check=False )
    CNNTrain( 2012, modelNum=103, check=False )
    
