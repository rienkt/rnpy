#!/usr/bin/env python
"""

  RESAMPLE.py

  resample 


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import scipy.signal as sp_signal
import rn.libs.fourier as rn_f


#======================================================================
# classes / functions
#======================================================================

def resample1d_sp( din, dxin, dxout ) :
  nout = int( float( din.shape[-1] ) * dxin / dxout )
  print nout, din.shape
  return sp_signal.resample( din, nout, axis=-1 ), nout

def resample1d_fft( din, dxin, dxout ) :
  nout = int( float( din.shape[-1] ) * dxin / dxout )
  fd, f = rn_f( din, dt=dxin )
#  if dxin > dxout :
#
#
#  elif dxin < dxout :
#    fd = np.delete( fd, np_s[ nout:], axis=-1)
#    
  print nout, din.shape, fd.shape
  return np.fft.irfft( fd ), nout

def resample2d_sp( din, d1in, d2in, d1out, d2out ) :
  n1, n2 = din.shape
  x1in = np.arange( 0, n1, dtype=np.float )*d1in
  x2in = np.arange( 0, n2, dtype=np.float )*d2in
  x1out = np.arange( x1in[0], x1in[-1], d1out, dtype=np.float )
  x2out = np.arange( x2in[0], x2in[-1], d2out, dtype=np.float )
  
  f = interpolate.interp2d( x1in, x2in, din,  kind = 'linear' )
  return f( x1out, x2out )


