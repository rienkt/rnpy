#!/usr/bin/env python
from __future__ import print_function
"""

  shape.py


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process

#==============================================================================
#                                                                PARAMETERS
#==============================================================================
#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================

def circle( o1, o2, r, ddeg=1. ) :
  degs = np.arange( 0., 360. + ddeg, ddeg, dtype=np.float ) 
  rads = np.deg2rad( degs )
  return o1 + r * np.cos( rads ),  o2 + r * np.sin( rads )



