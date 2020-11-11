#!/usr/bin/env python
from __future__ import print_function
"""

  Madagascar utilities

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================



def get_dim(input):
    dim=0
    for i in range(1,10):
        if input.int('n'+str(i))==None:
            break
        else:
            dim=i
    return dim

