# -*- coding: utf-8 -*-
"""
Created on Sat Dec 9 22:15:30 2023

Collect Results on IIM models: simple CNN, ResNet50, and ViT

Display and plot final results which are included in the project paper

@author: Ellen Wang
"""

import sys
if( "D:\\Users\\junwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\junwang\\source\\py\\ICEIsing" )

import IIMSimul
import datetime
from IIM_CNN_Results import collect_CNN_res
from IIM_Resnet_Results import collect_Resnet_res
from IIM_ViT_Results import collect_ViT_res

collecting = False # set to True if collect and save results
plotting = True # set to True if plotting and printing results

if ( collecting ):
# collect CNN predicted parameters, save parameters and the lattice in ..\\IIM_CNN_Outputs\\result_{:s}_{:s}_rundate{:s}.json
# No need to run any more. Already ran once and results have been saved

    ### Simple CNN    
    collect_CNN_res(2022)
    # collect_CNN_res(2023)
    # collect_CNN_res(2012)
    
    ### ResNet50
    # collect_Resnet_res( year=2023 )
    # collect_Resnet_res( year=2012 )
    # collect_Resnet_res( year=2022 )

    ### ViT
    # collect_ViT_res( year=2023 )


#### load saved results, plot and print results for the paper, very fast
if( plotting ):
    
    ### Simple CNN    
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


    ### ResNet50
    # # load annual results of 2023
    IIMSimul.load_year(2023, rd=datetime.date(2024,8,13), CNN = False, Resnet=True, plot_err=True)
    
    # # Resnet extent plots: 2023 run with PF=False
    # IIMSimul.extent_avg_plot( 2023, rd=datetime.date(2024,8,13), CNN=False, Resnet=True, PF=False )
    
    # # day by day simulation & plot
    # IIMSimul.day_by_day(2023,8,1, 2023,8,16, rd=datetime.date(2024,8,13),
    #                     savefile=True, CNN=False, Resnet=True)
    # IIMSimul.day_by_day(2023,8,16, 2023,9,1, rd=datetime.date(2024,8,13),
    #                     savefile=True, CNN=False, Resnet=True)
    # IIMSimul.day_by_day(2023,10,16, 2023,11,1, rd=datetime.date(2024,8,13),
    #                     savefile=True, CNN=False, Resnet=True)

    # # load annual results of 2012
    # IIMSimul.load_year(2012, rd=datetime.date(2024,9,4), CNN = False, Resnet=True, plot_err=True)
    
    # # Resnet extent plots: 2012 run 
    # IIMSimul.extent_avg_plot( 2012, rd=datetime.date(2024,9,4), CNN=False, Resnet=True, PF=False )
    
    # # load annual results of 2022
    # IIMSimul.load_year(2022, rd=datetime.date(2024,9,19), CNN = False, Resnet=True, plot_err=True)
    
    # # Resnet extent plots: 2022 run 
    # IIMSimul.extent_avg_plot( 2022, rd=datetime.date(2024,9,19), CNN=False, Resnet=True, PF=False )
    
    ### previous actual starting from 16Aug2023; for projection extrapolation analysis
    ### previous starting from 16Aug2023
    # IIMSimul.project_future(2023, 8, 16, rd = datetime.date(2024,9,19), 
    #                         pf_date = datetime.date(2024,9,20), savefile = False, Resnet=True )
    # IIMSimul.plot_actual_pf()

    ### Starting from 16Aug2023 compare 1Sep2023->1Jan2024 prediction results with actual 
    ### 16Jun2023->16Aug2023 plots are based on 2023 ResNet parameters
    ### 1Sep2023->1Jan2024 plots are based on 22022 ResNet parameters
    # IIMSimul.extent_avg_pred_comp( 2023, rd=datetime.date(2024,8,13), 
    #                   pf_date = datetime.date(2024,9,20), print_res = True, Resnet=True )


    ### ViT
    # IIMSimul.load_year(2023, rd=datetime.date(2024,8,21), 
    #                     CNN = False, Resnet=False, ViT=True, plot_err=True)
    
    # ViT extent plots: 2023 run with PF=False
    # IIMSimul.extent_avg_plot( 2023, rd=datetime.date(2024,8,21), CNN=False, Resnet=False, ViT=True, PF=False )

    