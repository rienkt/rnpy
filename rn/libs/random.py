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
#import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process
from scipy.special import gamma as sp_gamma
#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================


def von_karman1d( k, nu, a, sigma ) :
    return sigma**2 * (2*np.sqrt(np.pi)*a) * (sp_gamma( nu + 0.5 ))**( nu + 0.5 ) / sp_gamma(nu) / (1 + k**2 * a **2 )

