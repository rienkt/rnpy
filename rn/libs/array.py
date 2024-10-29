#!/usr/bin/env python
from __future__ import print_function
"""

  ARRAY.py

  array operation


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Kamei (Kamei)"

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
#                                                        CLASSES / FUNCTIONS
#==============================================================================

def dim_to2D( din ) :
  ndim      = din.ndim 
  nin       = din.shape[-1]
  nshape_in = list( din.shape ) 
  if ndim == 1 :
    din = np.reshape( din, ( 1, nin ) )
  elif ndim > 2 :
    ntrace = din.size / nin
    din = np.reshape( din, ( ntrace, nin ) )
  
  return din, nshape_in

def dim_to1D( din ) :
  ndim      = din.ndim 
  nin       = din.size
  nshape_in = list( din.shape ) 
  din = np.reshape( din, (nin) )
  return din, nshape_in

def dim_tonD( din, nshape_in ) :
  return din.reshape( nshape_in )

def find_nearest_value( array, vals ) : 
  #if type(vals) is not np.ndarray :
  #  vals = np.asarray( vals )
  if type(vals) is np.ndarray :
    vals, nshape = dim_to1D( vals ) 
    nval = vals.shape[0]
    idxs = np.zeros( nval, dtype=int )
    for ival, val in enumerate( vals ) :
      idxs[ival] = np.argmin( np.abs( array-val ) )
    return array[idxs].reshape( nshape ), idxs.reshape( nshape )
  else :  
    val = vals
    idx =  np.argmin( np.abs( array-val ) )
    return array[idx], idx

def find_nearest_value_regular( array, vals ) : 
  o = array[0]
  d = array[1]-array[0]
  n = array.shape[0]

  if type(vals) is np.ndarray :
    idxs = np.round( ( vals-o ) /d ).astype( int )
    idxs[ idxs<0 ]  =0
    idxs = np.minimum( idxs, n )
    idxs = np.maximum( idxs, 0 )
    return  idxs*d + o, idxs
  else :
    val = vals
    idx = np.round( ( val-o ) / d ).astype( int )
    idx = min( idx, n )
    idx = max( idx, 0 )
    return idx*d + o, idx
