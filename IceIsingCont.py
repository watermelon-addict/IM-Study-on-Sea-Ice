# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:25:40 2023

2-dimensional Ising model with continuous spin values

Based on https://github.com/red-starter/Ising-Model

Revised to allow continuous spin value and intertia factor, and location dependent external field

External field is set to a linear function of x and y position: B = B0 + Bx * (x-x0) + By * (y-y0)

An inertia factor is included for which describes the energy dissipated to flip a spin value

@author: Ellen Wang
"""

import numpy as np
from math import exp
import random

class Lattice:
    def __init__(self, si, NX=60, NY=60, J=1, B=0, T=1, Bx=0, By=0 ):

        # Ising model with continuous spin values
        
        # Bx and By is the decay of B along x and y direction. This is to take into account 
        # the variation of change from land to polar 
        # initialize the lattice: 1 is water, -1 is ice
        self.NX = NX
        self.NY = NY
        self.J = J
        self.B = B
        self.Bx = Bx
        self.By = By
        self.T = T
        self.E = 0 # total energy
        
        self.lat = np.zeros((self.NX, self.NY),dtype=float)

        random.seed(-12345678)

        for x in range(NX):
            for y in range( NY):
                self.lat[x,y] = si[x,y] * 2 - 1 # convert from 0 to 1 to -1 to 1

        self.E_tot()
        
    # calculates Hamiltonian for one cell 
    def E_elem(self,x,y, r = -999. ):

        # if r takes the default value, called from calculating total lattice energy
        if( r < -1.001 ) :
            curr = self.lat[x,y]
        # if flips spin value, called from metropolis
        else:
            curr = r

        # instead of cyclic lattice, we assume the out of bound cells have the same value as boundary
        # this is a reasonable assumption for the sea ice distribution
        if(x==0):
            upper = curr
        else:
            upper = self.lat[x-1,y]

        if(x==self.NX-1):
            lower = curr
        else:
            lower = self.lat[x+1,y]

        if(y==0):
            left = curr
        else:
            left = self.lat[x,y-1]

        if(y==self.NY-1):
            right = curr
        else:
            right = self.lat[x,y+1]
            
        # x is row, y is column, 
        # lower x closer to Russia; higher x closer to canada
        # lower y closer to land (Alaska); hiher y closer to polar
        # B is the external field at the center of the lattice
        ## B-Bx/2-By/2 is starting B at the top left corner
        BB = self.B - self.Bx/2 - self.By/2 + self.Bx * x / (self.NX-1) + self.By * y / (self.NY-1)       
              
        # the pairs of J * sigma_i * sigma_j are double counted for the full lattiace energy
        # But this does not matter for kinetic IM, because what's important is the change of energy in each metropolis step, which is correct        
        # We never use E_tot in this research.
        # But if really need E_tot, just set J to J/2 to remove double counting
        e = - self.J * curr *( upper + lower + left + right )

        if self.B != 0: # with magnetic field
            e += -1.0 * curr * BB
        
        return(e)                        

    # sums up Hamiltonian of full lattice
    def E_tot(self):
        en = 0
        for x in range(self.NX):
            for y in range(self.NY):
                en += self.E_elem(x,y)
        self.E = en


    # single metropolis step to change a spin value.
    # probability of flip is:
    # 100% if E' - E + inertia * abs( spin' - spin ) < 0
    # otherwise, exp( - beta * ( E' - E + inertia * | sigma' - sigma | ))
    def metropolis(self, steps = 1000, inertia = 1. ):
        for i in range(steps):
            # choose random atom
            x = random.randrange(self.NX)
            y = random.randrange(self.NY)

            E1 = self.E_elem(x,y)            
            r = random.uniform(-1, 1)
            
            E2 = self.E_elem(x,y,r)            
            dE = E2 - E1   
            dsigma = abs( r - self.lat[x,y])
            
            # inertia describes how hard to change from water to ice or vice versa
            # can be thought as latent heat for phase transition, or simply as inertia to any change from current state
            prob = exp(-1.0 * ( dE + inertia * dsigma ) / self.T )

            if ( prob >= 1 ):
                self.lat[x,y] = r
                self.E += dE                
            elif ( random.random() < prob ):
                self.lat[x,y] = r
                self.E += dE
