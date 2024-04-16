# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 13:10:22 2023

Simulation and plot functions for Sea Ice Continuous Ising Model

These functions are based on IceIsingCont.py and ReadSeaIce.py

They are called by IIMCNNModel.py and Gen_Figures.py

Functions directly used for the paper:
    IIM_period_test():	Tests a period with certain IM parameters and isplays images.
	read_result():		Reads Ising model simulation result
	load_year():		Loads a full year of results and plots.
	day_by_day(): 		Displays daily simultion vs. observation
	extent_avg_plot():	Displays the avg and extent of observations and simulations. argument "pf_date" is only used for Aug2023 project future.

Some functions in this script are used for intermediate results or debugging purpose, which include:        			project_future(): 	Test in Aug2023 to predict ice extent for the following months of 2023
	extent_avg_pred_comp():	Compare Aug2023 prediction vs. observations of Sep/Oct/Nov 2023.
	extent_avg_prev_years():Auxiliary functions to plot extent avg across different years
	IIM_cost_diff():    Calulation for target cost function for dual_annealing optimization.	
	IIM_period_cost():  Target cost function of a single period optimization.
	single_run():		Estimation of Ising parameters based for single priod using dual_annealing optimization.
	annual_run(): 		Estimation of Ising parameters based for all periods of a year in parallel.

@author: Ellen Wang
"""

import multiprocessing as mp
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing
import json
import datetime
from datetime import date
import matplotlib.dates as mdates
import time

import sys
if( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" not in sys.path ): 
    sys.path.append( "D:\\Users\\ellenwang\\source\\py\\ICEIsing" )
import IceIsingCont as IIMC # Ice Ising Model Continuous
from ReadSeaIce import read_SI
from IIMConstants import v1, v2, NX, NY, metrosteps, NumParam

plt.rcParams[ "font.family"] = "Times New Roman"

# auxiliary function to test a period with certain IM parameters (J, B, Bx, By, I)
# displays images of initial, simulations, and target
def IIM_period_test( params, args, iters = 1, plt_title = '' ):
    # iters is number of simulation states to be displayed
    
    # IM paramters
    j_is, b_is, bx, by, inertia = params

    # s1 is initial state, s2 is target state
    # s1 & s2 values are between 0 and 1; Ising lattice between -1 and 1
    s1, s2, NX, NY = args   

    
    ice_im = IIMC.Lattice(s1, NX = NX, NY = NY, J = j_is, B = b_is, Bx = bx, By = by )

    # iters+2 images will be displayed
    # first image is initial; final image is target
    # the middle iters images are simulation results
    # if iters = 1, then dispaly initial, simulation, and target state
    fig, axes = plt.subplots(iters+2,1)
    pos = axes[0].imshow(s1,cmap="Blues", vmin=v1, vmax=v2)
    fig.colorbar(pos, ax = axes[0])
    axes[0].axis("off")
    
    steps = int( metrosteps / iters ) # intermediate simulation steps
    
    # intermediate simulation outputs: if iters=1, then this includes 1 simulation result
    s1_outs = [] 
    for i in range(iters):
        ice_im.metropolis( steps, inertia )
        s1_out = [ 0.5 + x / 2. for x in ice_im.lat ]
        s1_outs.append(np.copy(s1_out))        

    # show simulation results 
    for i in range(iters):
        pos = axes[i+1].imshow(s1_outs[i],cmap="Blues", vmin=v1, vmax=v2)
        fig.colorbar(pos, ax = axes[i+1])
        axes[i+1].axis("off")

    tot_diff = IIM_cost_diff( s1_out, s2 )

    # show final target state
    pos = axes[iters+1].imshow(s2,cmap="Blues", vmin=v1, vmax=v2)
    fig.colorbar(pos, ax = axes[iters+1])
    axes[iters+1].axis("off")
    
    axes[0].set_title( plt_title)
    
    return( tot_diff, s1_outs )


# calculate difference of 2 lattices, used for optimization, not CNN
def IIM_cost_diff( s1_out, s2 ):

    # hyperparameters in the cost function 
    # cost function = 
    # difference of average of full lattiace ** 2 * alpha
    # +
    # ( sum of ( individual diff ** gamma ) / NX / NY ** (1/gamma) ) ** 2 ** beta 
    # we set alpha = 0, beta = 1, gamma = 1 so the result is simply the average of individual site difference - the manhattan distance
    alpha = 0  # weight of difference of average of the full lattice
    beta = 1 # weight of individual site difference 
    gamma = 1 # exponential factor on individual site difference: 1 is manhattan distance, 2 is Euclid distance

    a1 = np.average(s1_out)
    a2 = np.average(s2)

    s_diff = abs( s2 - s1_out )
    sd_diff = 0 # sum of individual differences
 
    for x in range( NX):
        for y in range(NY):
            sd_diff += s_diff[x,y] ** gamma
    sd_diff = ( sd_diff / NX / NY ) ** (1/gamma)
    a_diff = abs( a1 - a2 )
    
    tot_diff = a_diff ** 2 * alpha + sd_diff ** 2 * beta
    tot_diff = tot_diff ** 0.5 # convert back to first oder difference
    return( tot_diff )


# Ising run for a period; calculate the cost function value, which will be used as input for optimization func, not CNN
def IIM_period_cost( params, args ):
    # IM paramters
    j_is, b_is, bx, by, inertia = params

    # s1 is initial state, s2 is target state
    # s1 & s2 values are between 0 and 1; Ising lattice between -1 and 1
    s1, s2, NX, NY = args   
    
    # initialize ising lattice
    ice_im = IIMC.Lattice(s1, NX = NX, NY = NY, J = j_is, B = b_is, Bx = bx, By = by )

    # run a MC period
    ice_im.metropolis( metrosteps, inertia )

    # convert the output lattice from -1/1 to 0/1 to compare with target state s2
    s1_out = [ 0.5 + x / 2. for x in ice_im.lat ]

    tot_diff = IIM_cost_diff( s1_out, s2 )
    return( tot_diff )



# bounds for IM parameter minimization, for dual annealing optimization
bnds0 = [(2.25,2.75), (0,15), (-10,10), (-10,10), (9,11)]

# single period optimization using dual annealing
# this saves a json file with best fit parameters and the simulation output image
# this also show the image by calling IMM_period_test
def single_run( y1, m1, d1, y2, m2, d2, bnds = bnds0, savefile = True ):
    # freezing cycle starts from 16Sep-1Oct, ends at 1mar-16mar, so it's freezing cycle if target month in 10,11,12,1,2,3
    if( m2>=10 or m2<=3 ):
        bnds[1] = (-bnds[1][1], bnds[1][0]) 
    
    s1 = read_SI( y1, m1, d1)
    s2 = read_SI( y2, m2, d2)
    ret = dual_annealing(IIM_period_cost, bounds = bnds, args = [(s1, s2, NX, NY )] )
    print( "start", y1, m1, d1 )
    print( "end", y2, m2, d2 )
    print( ret.x )
    print( ret.fun )    
    d, s1_outs = IIM_period_test(ret.x, args = [s1, s2, NX, NY])

    if(savefile):        
        ymd = "{:4d}{:02d}{:02d}"
        ymd1 = ymd.format(y1, m1, d1)
        ymd2 = ymd.format(y2, m2, d2)
        tod = str(date.today())
        of = '..\\Outputs\\result_{:s}_{:s}_rundate{:s}.json'
        of = of.format( ymd1, ymd2, tod )    
        
        l = s1_outs[0].tolist()
        res = {"start": (y1,m1,d1), 
               "end": (y2,m2,d2), 
               "metrosteps": metrosteps,
               "retx": ret.x.tolist(),
               "retfun": ret.fun,
               "success": ret.success,
               "array": l}
        with open( of, "w") as outfile:
            json.dump( res, outfile, indent = 4)    

    return( ( ret, s1_outs) )


# parallel optimization runs for a full year
def annual_run( year ):
    start = time.time()

    bnds = bnds0
    ymd1 = [] # period start date
    ymd2 = [] # period end date

    curr_year = 2024 # when 2023 results are fully ready we chanage this to 2024
    # full 2023 data available now upon 4Jan2024

    # starts from 16Jun-1Jul period; ends 16Dec-1Jan period
    if( year < curr_year ):
        endmonth = 13
    else:
        # Run in August 2023
        # endmonth = 9 # 2023 only available up to 16Aug 
        # Now run in September 2023
        endmonth = 12 # 2023 only available up to 1Dec, as of december 2023
        
    for m in range( 7, endmonth):
        ymd2.append((year,m,1))
        ymd2.append((year,m,16))
        ymd1.append((year,m-1,16))
        ymd1.append((year,m,1))
    
    if( year < curr_year ):
        ymd2.append((year+1,1,1))
        ymd1.append((year,12,16))
    else:   # 2023 adding last available period 16Nov -> 1Dec
        ymd2.append((year,12,1))
        ymd1.append((year,11,16))
        
    sz = len(ymd1)
    th=[]
    for i in range(sz):
        y2,m2,d2=ymd2[i]
        y1,m1,d1=ymd1[i]
        th.append( mp.Process( target=single_run, args = (y1,m1,d1,y2,m2,d2,bnds) ) )

    for i in range(sz):
        th[i].start()

    for i in range(sz):
        th[i].join()

    end = time.time()
    print("execution time is:%6.0f seconds" % (end-start) )


# read best fit parameters and simulation image from saved results
def read_result(y1,m1,d1,y2,m2,d2, rd=date.today(), plot = False, printres = False, CNN=False ):
    ymd = "{:4d}{:02d}{:02d}"
    ymd1 = ymd.format(y1, m1, d1)
    ymd2 = ymd.format(y2, m2, d2)

    if( CNN ):
        of = '..\\IIM_CNN_Outputs\\result_{:s}_{:s}_rundate{:s}.json'
    else:      
        if( rd < datetime.date( 2023, 8, 20)): # different output results naming
            of = '..\\Outputs\\result_fixJL_{:s}_{:s}_rundate{:s}.json'
        else:
            of = '..\\Outputs\\result_{:s}_{:s}_rundate{:s}.json'

    of = of.format(ymd1, ymd2, str(rd) )    

    with open( of, "r") as openfile:
        jso = json.load(openfile)
    
    if( printres ):
        print(jso["start"], jso["end"], jso["success"], "metrosteps", jso["metrosteps"])
        for i in range( len(jso["retx"])):
            print(jso["retx"][i])
        print(jso["retfun"])
    
    a = np.array( jso["array"])

    if(plot):
        fig, axes = plt.subplots(1,1)
        pos = axes.imshow(a,cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(pos, ax = axes)
    
    if( rd < datetime.date(2023,10,1) ): # J was scaled to half after Oct2023
        jso["retx"][0] /= 2

    return( a, jso["retx"], jso["retfun"] )


# load results for a full year, plot the images and print best fit parameters
def load_year( year, rd = date.today(), CNN = False ):
    ymd1 = [] # period start date
    ymd2 = [] # period end date

    curr_year = 2024 # when 2023 results are fully ready we chanage this to 2024
    # full 2023 data available now upon 4Jan2024


    # starts from 16Jun-1Jul period; ends 16Dec-1Jan period
    if( year < curr_year ):
        endmonth = 13
    else:
        # Run in August 2023
        # endmonth = 9 # 2023 only available up to 16Aug
 
        # Now run in September 2023
        endmonth = 12 # 2023 only available up to 1Dec

        
    for m in range( 7, endmonth):
        ymd2.append((year,m,1))
        ymd2.append((year,m,16))
        ymd1.append((year,m-1,16))
        ymd1.append((year,m,1))
    
    if( year < curr_year ):
        ymd2.append((year+1,1,1))
        ymd1.append((year,12,16))
    else:   # 2023 adding last available period 16Nov -> 1Dec
        ymd2.append((year,12,1))
        ymd1.append((year,11,16))
        
    # actual holds all actual images; simualted holds simulated results
    # both start with initial actual images
    y1,m1,d1 = ymd1[0]
    s1 = read_SI( y1, m1, d1)
    actual = [s1]
    simulated = [s1]
    
    sz = len(ymd1) # number of periods
    retxs = []
    retfuns = []
    for i in range(sz):
        y2,m2,d2=ymd2[i]
        y1,m1,d1=ymd1[i]
        s2_actu = read_SI(y2,m2,d2)
        s2_simu, retx, retfun = read_result(y1,m1,d1,y2,m2,d2, rd = rd, CNN = CNN)
        actual.append(s2_actu)
        simulated.append(s2_simu)
        retxs.append(retx)
        retfuns.append(retfun)
    
    # plot the actual images first, 5 per row
    # then plot the simulated images, 5 per row
    NJ=5 # 5 images per row
    NI = (sz-1) // NJ + 1 # number of rows
    if( NI == 1):
        NI += 1 # temporarily making this 2 rows, will be automated after Sep results

    fig, axes = plt.subplots(NI,NJ)
    fig.subplots_adjust(right=0.95, wspace=0.00)

    for i in range(NI):
        for j in range(NJ):
            axes[i][j].axis('off')
            if( i*NJ+j < sz+1 ): # nubmer of images = sz+1 
                pos = axes[i][j].imshow(actual[i*NJ+j],cmap="Blues", vmin=v1, vmax=v2)
                label = chr( ord('a')+i*NJ+j)
                axes[i][j].set_title(f'({label})', y = -0.22) # add sub-figure label (a), (b), ...

    # add color bars
    # if( year < 2023 ):
        sub_ax = fig.add_axes([0.95, 0.66, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.95, 0.39, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.95, 0.125, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
    # else:
    #     sub_ax = fig.add_axes([0.96, 0.59, 0.01, 0.243])
    #     fig.colorbar(pos,cax=sub_ax)
    #     sub_ax = fig.add_axes([0.96, 0.2, 0.01, 0.243])
    #     fig.colorbar(pos,cax=sub_ax) 
        
    fig, axes = plt.subplots(NI,NJ)
    fig.subplots_adjust(right=0.95, wspace=0.00)
    for i in range(NI):
        for j in range(NJ):
            axes[i][j].axis('off')
            if( i*NJ+j < sz + 1 ): # nubmer of images = sz+1  
                pos = axes[i][j].imshow(simulated[i*NJ+j],cmap="Blues", vmin=v1, vmax=v2)
                label = chr( ord('a')+i*NJ+j)
                axes[i][j].set_title(f'({label})', y = -0.22) # add sub-figure label (a), (b), ...

    # add color bars
    # if( year < 2023 ):
        sub_ax = fig.add_axes([0.95, 0.66, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.95, 0.39, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.95, 0.125, 0.01, 0.22])
        fig.colorbar(pos,cax=sub_ax)
    # else:
    #     sub_ax = fig.add_axes([0.96, 0.59, 0.01, 0.243])
    #     fig.colorbar(pos,cax=sub_ax)
    #     sub_ax = fig.add_axes([0.96, 0.2, 0.01, 0.243])
    #     fig.colorbar(pos,cax=sub_ax)

    # print table of parameters, comma separated
    print( "param", end = ", " )
    for i in range(sz):
        ymd = "{:4d}{:02d}{:02d} to {:4d}{:02d}{:02d}"
        ymd = ymd.format(ymd1[i][0], ymd1[i][1], ymd1[i][2], ymd2[i][0], ymd2[i][1], ymd2[i][2])
        print( ymd, end = ", " )

    labels = ["J", "B0", "Bx", "By", "I"]
    for j in range( len(labels) ):
        print( "\n%s" % labels[j], end = ", ")
        for i in range(sz):
            print( retxs[i][j], end = ", " )

    print( "\n%s" % "retfun", end = ", ")
    for i in range(sz):
        print( retfuns[i], end = ", " )

    print( "\n" )
    return()


# simulate day by day for a semimonthly period, given Ising parameters
def day_by_day(y1,m1,d1,y2,m2,d2, rd=date.today(), savefile = True, plot = True, CNN=False ):
    # load best fit parameters
    s_simu, retx, retfun = read_result(y1,m1,d1,y2,m2,d2, rd = rd, CNN=CNN)    

    # divide metropolis simulation steps evenly for each day
    date1 = datetime.date(y1,m1,d1)
    date2 = datetime.date(y2,m2,d2)    
    delta = date2 - date1
    numdays = delta.days    
    daystep = metrosteps // numdays
    
    s1 = read_SI( y1, m1, d1)
    actual = [s1]
    simulated = [s1]
    
    j_is = retx[0]
    b_is = retx[1]
    bx = retx[2]
    by = retx[3]
    inertia = retx[4]
    
    ice_is = IIMC.Lattice(s1, NX = NX, NY = NY, J = j_is, B = b_is, Bx = bx, By = by )
    
    # simulate daily using the best-fit parameters
    for i in range(numdays):
        ice_is.metropolis( daystep, inertia )
        s1_out = [ 0.5 + x / 2. for x in ice_is.lat ]
        simulated.append(np.copy(s1_out))        
        
        # load daily actual
        date1 = date1 + datetime.timedelta( days = 1)
        yy = date1.year
        mm = date1.month
        dd = date1.day
        s2 = read_SI( yy, mm, dd)
        actual.append(s2)

    if(CNN):
        of = '..\\IIM_CNN_Outputs\\DayByDay_{:s}_{:s}_rundate{:s}.json'
    else:        
        of = '..\\Outputs\\DayByDay_{:s}_{:s}_rundate{:s}.json'

    if(savefile):        
        ymd = "{:4d}{:02d}{:02d}"
        ymd1 = ymd.format(y1, m1, d1)
        ymd2 = ymd.format(y2, m2, d2)
        tod = str(rd)
        of = of.format(ymd1, ymd2, tod )    
        
        res = {"start": (y1,m1,d1), 
               "end": (y2,m2,d2), 
               "metrosteps": metrosteps,
               "retx": retx,
               "actual": [],
               "simulated": []
               }
        
        for i in range(numdays+1):
            l = actual[i].tolist()
            res["actual"].append(l)
            ll = simulated[i].tolist()
            res["simulated"].append(ll)
                        
        with open( of, "w") as outfile:
            json.dump( res, outfile, indent = 4)    

    if( plot ):
        # each semimonth has 16-17 days, so the plot is 4 rows, 5 per row
        NI=4
        NJ=5

        fig, axes = plt.subplots(NI,NJ,figsize=(10,8))
        fig.subplots_adjust(right=0.95, wspace=0.00)
        for i in range(NI):
            for j in range(NJ):
                axes[i][j].axis('off')
                if( i*NJ+j < numdays+1 ):
                    pos = axes[i][j].imshow(actual[i*NJ+j],cmap="Blues", vmin=v1, vmax=v2)
                    label = chr( ord('a')+i*NJ+j)
                    axes[i][j].set_title(f'({label})', y = -0.18)
        # fig.suptitle( "Daily actual sea ice map")
        sub_ax = fig.add_axes([0.94, 0.717, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, 0.52, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, 0.323, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, .125, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
    
        fig, axes = plt.subplots(NI,NJ, figsize=(10,8))
        fig.subplots_adjust(right=0.95, wspace=0.00)
        for i in range(NI):
            for j in range(NJ):
                axes[i][j].axis('off')
                if( i*NJ+j < numdays+1 ):
                    pos = axes[i][j].imshow(simulated[i*NJ+j],cmap="Blues", vmin=v1, vmax=v2)
                    label = chr( ord('a')+i*NJ+j)
                    axes[i][j].set_title(f'({label})', y = -0.18)
        # fig.suptitle( "Daily evolution based on semi-monthly fitted params")
        sub_ax = fig.add_axes([0.94, 0.717, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, 0.52, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, 0.323, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, .125, 0.01, 0.163])
        fig.colorbar(pos,cax=sub_ax)

    return()


# project 2023 future ICE starting from 16Aug2023 to 1Jan2024
# based on best fit parameters from same period of 2022
# this is only used in Aug 2023 to test the model, its not used once 2023 full year data is available
def project_future(y0=2023, m0=8, d0=16, rd = date.today(), 
                   pf_date = datetime.date(2023,9,20), savefile = True, plot = True ):
    s1 = read_SI( y0, m0, d0)
    s_in = np.copy(s1)
    simulated = [s1]
    end_dates = [(y0,m0,d0)] # end dates for each simulation

    # period start and end dates, to be moved forward semimonthly every step
    y2 = y1 = y0
    m2 = m1 = m0
    d2 = d1 = d0

    while( y2 < 2024 ):    
        # move dates forward semi-monthly. Dates are always 1st and 16th of every month
        if( d2 == 1):
            d2 = 16
        else:
            d2 = 1 
            if( m2 == 12):
                y2 += 1
                m2 = 1
            else:
                m2 += 1

        # read best fit IM paramters from previous year
        s2_simu, retx, retfun = read_result(y1-1,m1,d1,y2-1,m2,d2, rd = rd)        
        end_dates.append((y2,m2,d2))
        y1 = y2
        m1 = m2
        d1 = d2
        # print( y2, m2, d2 )
        # print( retx, end = "\n\n" )
        
        # simulate one period forward based on previous year parameters
        ice_is = IIMC.Lattice(s_in, NX = NX, NY = NY, J = retx[0], B = retx[1], Bx = retx[2], By = retx[3] )
        ice_is.metropolis( metrosteps, retx[4] )
        s_out = [ 0.5 + x / 2. for x in ice_is.lat ]

        # save semimonthly simulated resutls for later outputs
        simulated.append(np.copy(s_out))        
        s_in = np.copy(s_out) # set simulated as new start for next period
        
    sz = len( simulated )
    # save simualted lattices to json
    # start with the initial lattice which will be saved too
    if(savefile):        
        ymd = "{:4d}{:02d}{:02d}"
        ymd1 = ymd.format(y0, m0, d0)
        rds = ymd.format( pf_date.year, pf_date.month, pf_date.day)
        of = '..\\Outputs\\ProjectFuture_{:s}_rundate{:s}.json'
        of = of.format(ymd1, rds )    
        
        res = {"end date": (y2,m2,d2), 
               "semimonthforward": sz-1,
               "simulated": []
               }
        
        for i in range(sz):
            ll = simulated[i].tolist()
            res["simulated"].append((end_dates[i],ll))
                        
        with open( of, "w") as outfile:
            json.dump( res, outfile, indent = 4)    

    # plot simulated forward semimonthly, start from y0, m0, d0
    if( plot ):
        v1 = 0.
        v2 = 1.
        NJ=5
        NI=(sz-1)//NJ + 1

        fig, axes = plt.subplots(NI,NJ,figsize=(10,4))
        fig.subplots_adjust(right=0.95, wspace=0.00)
        
        for i in range(NI):
            for j in range(NJ):
                axes[i][j].axis('off')
                if( i*NJ+j < sz ):
                    pos = axes[i][j].imshow(simulated[i*NJ+j],cmap="Blues", vmin=v1, vmax=v2)
                    label = chr( ord('a')+i*NJ+j)
                    axes[i][j].set_title(f'({label})', y = -0.15)
        # fig.suptitle( "2023 Semi-monthly Forward Projection")

        sub_ax = fig.add_axes([0.94, 0.54, 0.01, 0.34])
        fig.colorbar(pos,cax=sub_ax)
        sub_ax = fig.add_axes([0.94, 0.13, 0.01, 0.34])
        fig.colorbar(pos,cax=sub_ax)

    return()


# plot average ice percentage and ice extent
# there are run dates, first is the best-fit parameter rund ate, 
# second pf_date is for 2023 proj future date (date of generating 2022 best-fit param), which is only needed because of 2023 actual are partial
# once we pass 2023, and we simulate the full 2023 results, then change curr_year to 2024
def extent_avg_plot( year, rd=date.today(), pf_date = datetime.date(2023,8,23), 
                    print_res = False, PF = True, CNN = False ):
    ymd1 = [] # period start date
    ymd2 = [] # period end date
    
    # now full 2023 data available as of 4Jan2024
    curr_year = 2024 # when 2023 results are fully ready we chanage this to 2024

    # starts from 16Jun-1Jul period; ends 16Dec-1Jan period
    if( year < curr_year ):
        endmonth = 13
    else:
        # Run in August 2023
        # endmonth = 9 # 2023 only available up to 16Aug
 
        # Now run in Dec 2023
        endmonth = 12 # 2023 only available up to 1Dec2023
        
    for m in range( 7, endmonth):
        ymd2.append((year,m,1))
        ymd2.append((year,m,16))
        ymd1.append((year,m-1,16))
        ymd1.append((year,m,1))
    
    if( year < curr_year ):
        ymd2.append((year+1,1,1))
        ymd1.append((year,12,16))
    else:   # 2023 adding last available period 16Nov -> 1Dec
        ymd2.append((year,12,1))
        ymd1.append((year,11,16))

    # start from first actual image
    y1,m1,d1 = ymd1[0]
    s1 = read_SI( y1, m1, d1)
    actual = [s1]
    simulated = [s1]
    
    # read in actual image and simulation results
    sz = len(ymd1)
    for i in range(sz):
        y2,m2,d2=ymd2[i]
        y1,m1,d1=ymd1[i]
        s2_actu = read_SI(y2,m2,d2)
        s2_simu, retx, retfun = read_result(y1,m1,d1,y2,m2,d2, rd=rd, CNN=CNN )
        actual.append(s2_actu)
        simulated.append(s2_simu)

    sz_act = len(actual)
    
    # unfortunately for 2023 we might have different numbers of periods of results for actual and simulations
    all_act_dates = ymd1.copy()
    all_act_dates.append(ymd2[-1])    
    days_act = []
    for i in range(sz_act):
        d = datetime.date(all_act_dates[i][0], all_act_dates[i][1], all_act_dates[i][2])
        days_act.append(d)
    days_sim = days_act.copy()
        
    # for 2023, we need to load the projected future images
    if( PF and year == curr_year ):
        # start date of 2023 projection is 16Aug2023
        # mm = 8
 
        # Now run in September 2023
        mm = 9

        dd = 16
        ymd = "{:4d}{:02d}{:02d}"
        ymd1 = ymd.format(year, mm, dd)
        pfs = ymd.format(pf_date.year, pf_date.month, pf_date.day)
        of = '..\\Outputs\\ProjectFuture_{:s}_rundate{:s}.json'
        of = of.format(ymd1, pfs )    
        
        with open( of, "r") as openfile:
            jso = json.load(openfile)
        
        sim = jso["simulated"]
        smf = len( sim ) # of simulation periods plus 1 in project_future
        
        dd = datetime.date(y2,m2,d2)
        # first one is 1Aug2023 actual, so we start with second one which is the simulation restuls
        for i in range( 1, smf ):    
            a = np.array(sim[i]) # a has the end date as first value, simulated lattice as second value
            simulated.append(a[1])
            days_sim.append(datetime.date(a[0][0],a[0][1],a[0][2])) # year month day of the simulation end date
        
    sz_sim = len(simulated)

    # We have all actual and simulated images and dates vectors, we can plot now!
    act_avgs = []
    sim_avgs = []
    act_exts = []
    sim_exts = []
    
    for i in range(sz_act):
        # avg is the average water percentage across the lattice
        # 1- to covnert back to ICE average
        act_avg = 1 - np.average(actual[i])
        act_avgs.append(act_avg)
        act_ext = np.zeros((NX, NY))
        for x in range(NX):
            for y in range(NY):
                if( actual[i][x][y] <= 0.85): # extent is defined if ice >= 15%
                    act_ext[x][y]=1
        act_exts.append( np.sum(act_ext) / (NX*NY) )


    for i in range(sz_sim):
        # avg is the average water percentage across the lattice
        # 1- to covnert back to ICE average
        sim_avg = 1 - np.average(simulated[i])
        sim_avgs.append(sim_avg)
        sim_ext = np.zeros((NX, NY))
        for x in range(NX):
            for y in range(NY):
                if( simulated[i][x][y] <= 0.85):
                    sim_ext[x][y]=1
        sim_exts.append( np.sum(sim_ext) / (NX*NY) )
        
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(days_act, act_avgs, label = 'actual')
    ax[0].plot(days_sim, sim_avgs, label = 'simulated')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter( '%Y-%m-%d'))
    ax[0].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    ax[0].set_xlim( datetime.date(y1,6,6), datetime.date(y1+1,1,11))
    fig.autofmt_xdate()
    ax[0].set_xlabel( '(a)')
    ax[0].set_ylabel( 'ICE Coverage Percentage')
    # ax[0].set_title( "Average ICE Coverage Over Time" )
    ax[0].legend( loc=9 ) # center top position

    ax[1].plot(days_act, act_exts, label = 'actual')
    ax[1].plot(days_sim, sim_exts, label = 'simulated')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter( '%Y-%m-%d'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    ax[1].set_xlim( datetime.date(y1,6,6), datetime.date(y1+1,1,11))
    fig.autofmt_xdate()
    ax[1].set_xlabel( '(b)')
    ax[1].set_ylabel( 'ICE Extent')
    # ax[1].set_title( "ICE Extent Over Time" )
    ax[1].legend( loc=9 )
    # plt.suptitle( "Average Ice Coverage Percentage and Extent over Time for %4d" % year)

    if( print_res ):
        print("days_act")
        print( days_act )
        print("days_sim")
        print( days_sim )
        print("actual average")
        print( act_avgs )
        print("simulated average")
        print( sim_avgs )
        print("actual extent")
        print( act_exts )
        print("simulated extent")
        print( sim_exts )

    return()


# plot average ice percentage and ice extent for 2023 prediction vs actual
# 16Aug2023 is the final data date that prediction was based on 
# again this is only to test the quality of the prediction done in Aug2023 
def extent_avg_pred_comp( year, rd=date.today(), pf_date = datetime.date(2023,8,9), print_res = True ):
    ymd1 = [] # period start date
    ymd2 = [] # period end date

    endmonth = 9 # prediction was based on 16Aug2023
        
    for m in range( 7, endmonth):
        ymd2.append((year,m,1))
        ymd2.append((year,m,16))
        ymd1.append((year,m-1,16))
        ymd1.append((year,m,1))


    # start from first actual image
    y1,m1,d1 = ymd1[0]
    s1 = read_SI( y1, m1, d1)
    actual = [s1]
    simulated = [s1]
    
    # read in actual image and simulation results
    sz = len(ymd1)
    for i in range(sz):
        y2,m2,d2=ymd2[i]
        y1,m1,d1=ymd1[i]
        s2_actu = read_SI(y2,m2,d2)
        s2_simu, retx, retfun = read_result(y1,m1,d1,y2,m2,d2, rd=rd)
        actual.append(s2_actu)
        simulated.append(s2_simu)


    # add newly availalbe actuals
    ymd1_new = []
    ymd2_new = [] 
    ymd1_new.append((2023,8,16))
    ymd1_new.append((2023,9,1))
    ymd1_new.append((2023,9,16))
    ymd1_new.append((2023,10,1))
    ymd1_new.append((2023,10,16))

    ymd2_new.append((2023,9,1))
    ymd2_new.append((2023,9,16))
    ymd2_new.append((2023,10,1))
    ymd2_new.append((2023,10,16))
    ymd2_new.append((2023,11,1))
    sz = len(ymd1_new)
    for i in range(sz):
        y2,m2,d2=ymd2_new[i]
        y1,m1,d1=ymd1_new[i]
        s2_actu = read_SI(y2,m2,d2)
        actual.append(s2_actu)

    sz_act = len(actual)    
    all_act_dates = ymd1.copy() + ymd1_new.copy()
    all_act_dates.append(ymd2_new[-1])    
    days_act = []
    for i in range(sz_act):
        d = datetime.date(all_act_dates[i][0], all_act_dates[i][1], all_act_dates[i][2])
        days_act.append(d)


    act_dates1 = ymd1.copy()
    act_dates1.append(ymd2[-1])    
    sz1 = len(act_dates1)
    days_sim = []
    for i in range(sz1):
        d = datetime.date(all_act_dates[i][0], all_act_dates[i][1], all_act_dates[i][2])
        days_sim.append(d)
        
    mm = 8
    dd = 16
    ymd = "{:4d}{:02d}{:02d}"
    ymd1 = ymd.format(year, mm, dd)
    pfs = ymd.format(pf_date.year, pf_date.month, pf_date.day)
    of = '..\\Outputs\\ProjectFuture_{:s}_rundate{:s}.json'
    of = of.format(ymd1, pfs )    
    
    with open( of, "r") as openfile:
        jso = json.load(openfile)
    
    sim = jso["simulated"]
    smf = len( sim ) # of simulation periods plus 1 in project_future
    
    dd = datetime.date(y2,m2,d2)
    # first one is 1Aug2023 actual, so we start with second one which is the simulation restuls
    for i in range( 1, smf ):    
        a = np.array(sim[i]) # a has the end date as first value, simulated lattice as second value
        simulated.append(a[1])
        days_sim.append(datetime.date(a[0][0],a[0][1],a[0][2])) # year month day of the simulation end date
        
    sz_sim = len(simulated)

    # We have all actual and simulated images and dates vectors, we can plot now!
    act_avgs = []
    sim_avgs = []
    act_exts = []
    sim_exts = []
    
    for i in range(sz_act):
        # avg is the average water percentage across the lattice
        # 1- to covnert back to ICE average
        act_avg = 1 - np.average(actual[i])
        act_avgs.append(act_avg)
        act_ext = np.zeros((NX, NY))
        for x in range(NX):
            for y in range(NY):
                if( actual[i][x][y] <= 0.85): # extent is defined if ice >= 15%
                    act_ext[x][y]=1
        act_exts.append( np.sum(act_ext) / (NX*NY) )


    for i in range(sz_sim):
        # avg is the average water percentage across the lattice
        # 1- to covnert back to ICE average
        sim_avg = 1 - np.average(simulated[i])
        sim_avgs.append(sim_avg)
        sim_ext = np.zeros((NX, NY))
        for x in range(NX):
            for y in range(NY):
                if( simulated[i][x][y] <= 0.85):
                    sim_ext[x][y]=1
        sim_exts.append( np.sum(sim_ext) / (NX*NY) )
        
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(days_act, act_avgs, label = 'actual')
    ax[0].plot(days_sim, sim_avgs, label = 'predicted')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter( '%Y-%m-%d'))
    ax[0].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    fig.autofmt_xdate()
    ax[0].set_xlabel( '(a)')
    ax[0].set_ylabel( 'ICE Coverage Percentage')
    # ax[0].set_title( "Average ICE Coverage Over Time" )
    ax[0].legend( loc=9 ) # center top position

    ax[1].plot(days_act, act_exts, label = 'actual')
    ax[1].plot(days_sim, sim_exts, label = 'predicted')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter( '%Y-%m-%d'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    fig.autofmt_xdate()
    ax[1].set_xlabel( '(b)')
    ax[1].set_ylabel( 'ICE Extent')
    # ax[1].set_title( "ICE Extent Over Time" )
    ax[1].legend( loc=9 )
    # plt.suptitle( "Average Ice Coverage Percentage and Extent over Time for %4d" % year)

    if( print_res ):
        print("days_act")
        print( days_act )
        print("days_sim")
        print( days_sim )
        print("actual average")
        print( act_avgs )
        print("simulated average")
        print( sim_avgs )
        print("actual extent")
        print( act_exts )
        print("simulated extent")
        print( sim_exts )

    return()


# auxilliary functions to plot extent avg across different years
def extent_avg_prev_years( years = [2012, 2023, 2019, 2020] ):
    ymds = {}
    for year in years:
        ymd = []
        # start from 16Jun, end 1Jan
        for m in range( 7, 13):
            ymd.append((year,m-1,16))
            ymd.append((year,m,1))
        # 2023 data after 1Nov2023 is not available; pop 2 times to remove 16Nov to 1Dec
        if( year == 2023):
            ymd.pop()
            ymd.pop()
        else:
            ymd.append((year, 12, 16))
            ymd.append((year+1, 1,1))
            
        ymds[year]=ymd

    actuals = {}
    for year in years:
        actual = []
        for ddd in ymds[year]:
            y,m,d=ddd
            actu = read_SI(y,m,d)
            actual.append(actu)
        actuals[year] = actual

    dates = {}
    for year in years:
        sz = len(actuals[year])    
        dd = []
        for i in range(sz):
            if i == sz-1 and year != 2023:
                y = 2024
            else:
                y=2023
            d = datetime.date(y, ymds[year][i][1], ymds[year][i][2]) # show only month and day
            dd.append(d)
        dates[year] = dd
        
    
    act_avgss = {}
    act_extss = {}
    
    for year in years:        
        act_avgs=[]
        act_exts=[]
        actual = actuals[year]
        for i in range(len(actual)):
            # avg is the average water percentage across the lattice
            # 1- to covnert back to ICE average
            act_avg = 1 - np.average(actual[i])
            act_avgs.append(act_avg)
            act_ext = np.zeros((NX, NY))
            for x in range(NX):
                for y in range(NY):
                    try:
                        if( actual[i][x][y] <= 0.85): # extent is defined if ice >= 15%
                            act_ext[x][y]=1
                    except:
                        print( "exception: ", year, dates[year][i], x, y)
            act_exts.append( np.sum(act_ext) / (NX*NY) )
        act_avgss[year] = act_avgs
        act_extss[year] = act_exts
        
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    for year in years:
        ax[0].plot(dates[year], act_avgss[year], label = str(year) )
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter( '%m-%d'))
    ax[0].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    fig.autofmt_xdate()
    ax[0].set_xlabel( '(a)')
    ax[0].set_ylabel( 'ICE Coverage Percentage')
    # ax[0].set_title( "Average ICE Coverage Over Time" )
    ax[0].legend( loc=9 ) # center top position

    for year in years:
        ax[1].plot(dates[year], act_extss[year], label = str(year) )
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter( '%m-%d'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator( interval=38 ))
    fig.autofmt_xdate()
    ax[1].set_xlabel( '(b)')
    ax[1].set_ylabel( 'ICE Extent')
    # ax[1].set_title( "ICE Extent Over Time" )
    ax[1].legend( loc=9 )
    # plt.suptitle( "Average Ice Coverage Percentage and Extent over Time for %4d" % year)

    return()

# debugging purpose only
# if ( __name__ == '__main__'): # usually do not run
#     annual_run(2023)

# simple testing
# load_year(2023, rd=datetime.date(2024,1,6))
