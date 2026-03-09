#!/usr/bin/env python
from __future__ import print_function
"""


  <MODULE NAME>

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
import h5py
#import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================


def PrintAllObjects(name) :
    print( name )

def PrintOnlyDataset( name, obj ) :
    if isinstance( obj, h5py.Dataset) :
        print( name, '\t', obj, '\t', obj[0] )

