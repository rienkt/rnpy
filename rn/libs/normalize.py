#!/usr/bin/env python




from __future__ import print_function
"""

  NORMALIZE.py

  normalize array


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np



#======================================================================
# classes / functions
#======================================================================

# normalize by max amplitude of specified axis
#between idx i0 and i1 
def normalize( d, axis=-1, i0=0, i1=None) :

  if axis >= 0 :
    d = np.swapaxes( d, -1, axis )

   


  nshape = d.shape
  n1 = d.shape[-1]
  n0 = int( d.size /n1  )
  print( d.size, n0, n1 )

  if i1 is None :
    i1 = n1 
  #print( d.size, n0, n1 )
  import copy
  dout = copy.copy( d) 
  dout = dout.reshape( ( n0, n1 ) )
 
  idx = np.where( np.isnan( dout ) )

  amax = np.amax( np.abs(dout[:,i0:i1] ), axis = -1 ) 
  #print( amax, amax.shape)

  dout[idx] = 0.

  amax[ amax < 1e-28 ] = 1. 

  for i0 in range( n0) :
    
    dout[ i0, : ] = dout[ i0, : ] / amax[ i0 ]
    #print( np.abs( d[i0,:] ).max())

    
  if axis >= 0 :
    return np.swapaxes( dout.reshape( nshape ), -1, axis )
  else :
    return dout.reshape( nshape )

def normalize_rms( d, axis=-1, i0=0, i1=None) :

  if axis >= 0 :
    d = np.swapaxes( d, -1, axis )

   


  nshape = d.shape
  n1 = d.shape[-1]
  n0 = d.size /n1 

  if i1 is None :
    i1 = n1 
  #print( d.size, n0, n1 )

  d = d.reshape( ( n0, n1 ) )
 
  idx = np.where( np.isnan( d ) )

  amax = np.sqrt( np.sum( np.abs(d[:,i0:i1] )**2, axis = -1 ) / (i1-i0) )

  d[idx] = 0.

  amax[ amax < 1e-28 ] = 1. 

  for i0 in range( n0) :
    d[ i0, : ] = d[ i0, : ] / amax[ i0 ]

  #amax = np.max( np.abs( d[ :, i0:i1] ) )
  #print(amax)
  #d /= amax
  #amax = np.max( np.abs( d[ :, i0:i1]  ) )
    
  if axis >= 0 :
    return np.swapaxis( d.reshape( nshape ), -1, axis )
  else :
    return d.reshape( nshape )
# d

#  amax = np.max( d, axis=axis ) 
#  nshape = d.shape
#  amax = amax.flatten()
#  ntrace = len(amax) 
#  nt     = d.shape[-1]
#
#  amax[ np.where( amax == 0 ) ] = 1.
#
#  for itrace in range( ntrace ) :
#    d[ itrace, : ] /= amax[ itrace ]
#
