# -*- coding: utf-8 -*-
"""
Created on Sat June 24 22:47:15 2023

Read NSIDC sea ice .nc file to a 2-D array; load our focus area and normalize values. 
This script also includes a function avg_freeze() which calculates the average ice level of the lattice.

@author: Ellen Wang
"""

from matplotlib import pyplot as plt
import numpy as np
import netCDF4
plt.rcParams[ "font.family"] = "Times New Roman"


def read_SI( year=2023, month=7, day=1 ):
    ymd = "{:4d}{:02d}{:02d}"
    ymd = ymd.format(year, month, day)

    # different detector data across periods
    if( year<1995):
        fp = '..\\NSIDC\\NSIDC0051_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F11_ICECON"
    elif( year<2010):
        fp = '..\\NSIDC\\NSIDC0051_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F13_ICECON"
    elif( year<=2020):
        fp = '..\\NSIDC\\NSIDC0051_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F17_ICECON"
    elif( year==2021 and month <=10):
        fp = '..\\NSIDC\\NSIDC0051_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F17_ICECON"
    elif( year==2022 and month == 8 and day >= 29):
        fp = '..\\NSIDC\\NSIDC0081_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F17_ICECON"
    else:
        fp = '..\\NSIDC\\NSIDC0081_SEAICE_PS_N25km_{:s}_v2.0.nc'
        key = "F18_ICECON"

    fp = fp.format(ymd)
    nc = netCDF4.Dataset(fp)
    var = nc.variables
    keys = nc.variables.keys()
    si = var[key]
    ss = si[0,180:240,80:140] # 60x60 betwwwn Beaufort sea, East Siberian Sea and Pole
    # ss = si[0,321:346,52:77] # hudson bay
    # ss = si[0,:,:] # full region
    
    # flip 0 and 1, so the output be 1 as water, 0 as ice, so plot is easier as "Blues"
    ss = [ 1-x for x in ss ]
    ss = np.array(ss)
    return(ss)

# calculate the average ice freeze level of the whole lattice: 0 is all water, 1 is all ice
def avg_freeze( ss, NX = 25, NY = 25, NS = 10):
    lat_small = np.zeros((NX,NY), dtype=float)
    for x in range(NX):
        for y in range( NY):
            lat_small[x,y] = np.average(ss[x*NS:x*NS+NS, y*NS:y*NS+NS])
            lat_small[x,y] += 0.5*(1-lat_small[x,y]) # convert from -1,1 to 0,1

    avg = np.average(lat_small)
    return( avg, lat_small)
    
if( __name__ == "__main__"):
    s0 = read_SI( 2023,8,16 )
    plt.imshow(s0,cmap="Blues", vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

