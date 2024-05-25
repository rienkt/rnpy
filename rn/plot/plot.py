#!/usr/bin/env python
from __future__ import print_function
"""


  rn/plot/plot.py

  some plot functions


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================

def rn_text( x, y, s, **kwargs ) :
  plt.txt( 
