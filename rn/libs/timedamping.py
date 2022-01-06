#!/usr/bin/env python
"""

  timedamping.py

  Apply time damping to data


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


def apply( d, t, laplace, tau=None ) :
  dshape = d.shape
  if d.ndim == 3 :
    d = d.reshape( dshape[0]*dshape[1], dshape[-1] )
  ntrace, nt = d.shape


  if tau is not None :
    laplace = 1./tau

  filt = np.exp( -t * laplace )
  #print filt.shape
  #print filt
  #print t
  #print laplace

  ffilt = np.repeat( filt.reshape( (1, nt) ), ntrace, axis=0 )

  #print ffilt.shape
  #print ffilt

  dout = ( d * ffilt ).reshape( dshape )

  #print dout.max()


  return dout


