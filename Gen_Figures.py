# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:15:30 2023

Collect Results on IIM CNN

Display and plot final results which are included in the project paper

@author: Ellen Wang
"""

import sys
if( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" )

import IIMSimul
import datetime
from IIM_CNN_Results import collect_CNN_res

collecting = False # set to True if collect and save results
plotting = True # set to True if plotting and printing results

if ( collecting ):
# collect CNN predicted parameters, save parameters and the lattice in ..\\IIM_CNN_Outputs\\result_{:s}_{:s}_rundate{:s}.json
# No need to run any more. Already ran once and results have been saved
    collect_CNN_res(2022)
    collect_CNN_res(2023)
    collect_CNN_res(2012)


#### load saved results, plot and print results for the paper, very fast
if( plotting ):    
    # load annual results of 2022, 2012 and 2023
    IIMSimul.load_year(2022, datetime.date(2023,12,10), CNN = True)
    IIMSimul.load_year(2023, datetime.date(2023,12,10), CNN = True)
    IIMSimul.load_year(2012, datetime.date(2023,12,10), CNN = True)
    
    # day by day simulation & plot
    IIMSimul.day_by_day(2022,8,16, 2022,9,1, rd=datetime.date(2023,12,10),savefile=False,CNN=True)
    IIMSimul.day_by_day(2022,10,16, 2022,11,1, rd=datetime.date(2023,12,10),savefile=False,CNN=True)
    
    # CNN extent plots: 2023 run with PF=False
    IIMSimul.extent_avg_plot( 2022, rd=datetime.date(2023,12,10), CNN=True )
    IIMSimul.extent_avg_plot( 2023, rd=datetime.date(2023,12,10), CNN=True, PF=False )
    IIMSimul.extent_avg_plot( 2012, rd=datetime.date(2023,12,10), CNN=True )
    