#!/usr/bin/env python
from __future__ import print_function
"""

  CONVOLVE.py

  Convolution


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Kamei (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process
import rk.libs.array as rk_array

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================


# convolution is comulative
# start by assuming 2D 
def convolve( d1, d2, axis=-1 ) :
  d1, nshape1 = rk_array.dim_to2D( d1 )
  d2, nshape2 = rk_array.dim_to2D( d2 )

  n1_0, n1_1 = d1.shape
  n2_0, n2_1 = d2.shape

  if n1_1 != n2_1 :
    print( 'second dimension of two data needs to be the same ' )
    return
  else :
    n1 = n1_1


  if n1_0 < n2_0 :
    n0 = n2_0 
    # only use the first trace 
    d1 = np.repeat( d1, n0 ).reshape( n1, n0 ).T
  elif n2_0 < n1_0 :
    n0 = n1_0
    d2 = np.repeat( d2, n0 ).reshape( n1, n0 ).T


  fd1 = np.fft.rfft( d1 )
  fd2 = np.fft.rfft( d2 )

  return np.fft.irfft( fd1*fd2 )

  #return sp.signal.fftconvolve( d1, d2,  mode='same', axes=1 )
  #return np.convolve( d1, d1, mode='same' )





