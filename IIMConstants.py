# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:23:20 2023

Constants used in Ice Ising model

@author: Ellen Wang
"""

# lattice size
NX = 60
NY = 60

# hyperparameter of Ising model metropolis steps for each semi-monthly period
metrosteps = 50000

# number of Ising parameters: J, B0, Bx, By, I
NumParam = 5

# image color bar range: here shows full range between 0 and 1
v1 = 0.
v2 = 1.