#!/usr/bin/env python
"""

  LMO.py

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np

import rn.libs as rn_lib
import rn.libs.fourier as rn_f

#======================================================================
# classes / functions
#======================================================================



# calculate reduced time
def lmo( d, dt, tshift, ot=0., otshift=0. ) :
  nshape = d.shape
  nt = d.shape[-1]
  ntrace = int( d.size / nt )
  print( nshape, nt, ntrace )
  d = d.reshape( ntrace, nt )
  tshift = tshift.reshape( ntrace )
  #print( np.arange( nt, dtype=np.float ) *dt )
  fd, f = rn_f.rfft( d, t=np.arange( nt, dtype=float )*dt )
  ntrace = d.shape[0]
  nf = f.shape[-1]
  ttshift = np.repeat( tshift.reshape( ntrace, 1 ), nf, axis=-1 )

  fd *= np.exp( 2j * np.pi * f * ( ttshift + otshift )  )

  dout = np.fft.irfft( fd, axis=-1 ).reshape( nshape )
  tout = np.arange( nt, dtype=float )*dt + otshift + ot


  return dout, tout

def time_phase_shift( d, dt, tshift, phzshift ) :
  nt = d.shape[-1] 
  ntrace = d.shape[0]
  fd = np.fft.rfft( d, axis=-1 )
  f = np.arange( 0, nt/2+1, dtype=float ) / dt / float(nt) 
  nf = f.shape[-1]
  ttshift = np.repeat( tshift.reshape( ntrace, 1 ), nf, axis=-1 )
  pphzshift = np.repeat( phzshift.reshape( ntrace, 1 ), nf, axis=-1 )

  fd *= np.exp( -2j * np.pi * f * ttshift + 1j*pphzshift)

  dout = np.fft.irfft( fd, axis=-1 )
  return dout

 



