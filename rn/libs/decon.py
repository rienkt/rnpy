#!/usr/bin/env python
from __future__ import print_function
"""
 RK_DECON.PY variables < infile > outfile

  Deconvolution


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import sys
import rk.libs.fourier as rk_fourier

#======================================================================
# Parameters
#======================================================================



#======================================================================
# Functions
#======================================================================

#======================================================================
# Main
#======================================================================

def decon( ref, child, damp=0.002) :

  # ref and child needs to be the same size
  # decon along the last axis

  nshape = ref.shape
  n1 = ref.shape[-1]
  n0 = ref.size / n1

  fd0 = np.fft.fft( ref ).reshape( ( n0, n1 ) )
  fd1 = np.fft.fft( child ).reshape( ( n0, n1 ) )
 
  amax_fd0 = np.max( np.abs( fd0 ) , axis=-1 )
  amax_fd1 = np.max( np.abs( fd1 ) , axis=-1 )

  damp_fd0 = np.repeat( amax_fd0.reshape( n0, 1 )**2, n1, axis=1 ) * damp
  damp_fd1 = np.repeat( amax_fd1.reshape( n0, 1 )**2, n1, axis=1 ) * damp


  fd0 = np.ma.masked_equal( fd0, damp_fd0 == 0. )
  fd1 = np.ma.masked_equal( fd1, damp_fd1 == 0. )

  fc  = fd1 * np.conj( fd0 ) / ( np.abs( fd0 ) **2 + damp_fd0 )
  return np.real( np.fft.ifftshift( np.fft.ifft( fc ) , axes=-1) ).reshape( nshape )

def decon_remove_wavelet( ref, child, damp=0.002, pad=0., otref=0.,
                          dt=1., lowcut=None, highcut=None ) :

  # ref and child needs to be the same size
  # ref is one trace
  # child can have many traces
  # decon along the last axis


  nshape = child.shape
  n1_ref = child.shape[-1]
  n0 = child.size / n1_ref

  ref = np.repeat( ref, n0 ).reshape( n1_ref, n0 ).T
  #print( 'ref', ref[ 0, :] )

  if pad > 0 :
    pad = np.zeros( ( n0, int(pad*n1_ref) ), dtype=np.float )
    ref = ref.reshape( ( n0, n1_ref ) )
    child = child.reshape( ( n0, n1_ref ) )
    ref = np.concatenate( ( ref, pad ), axis=-1 )
    child = np.concatenate( ( child, pad ), axis=-1 )
    n1 = ref.shape[-1]
  else :
    n1 = n1_ref


  fd0 = np.fft.fft( ref ).reshape( ( n0, n1 ) )
  fd1 = np.fft.fft( child ).reshape( ( n0, n1 ) )
 
  amax_fd0 = np.max( np.abs( fd0 ) , axis=-1 )
  amax_fd1 = np.max( np.abs( fd1 ) , axis=-1 )

  damp_fd0 = np.repeat( amax_fd0.reshape( n0, 1 )**2, n1, axis=1 ) * damp
  damp_fd1 = np.repeat( amax_fd1.reshape( n0, 1 )**2, n1, axis=1 ) * damp


  fd0 = np.ma.masked_equal( fd0, damp_fd0 == 0. )
  fd1 = np.ma.masked_equal( fd1, damp_fd1 == 0. )

  fc  = fd1 * np.conj( fd0 ) / ( np.abs( fd0 ) **2 + damp_fd0 )

  dout = np.roll( np.real( np.fft.ifft( fc )  ), -int(otref/dt), axis=-1 )[ :, :n1_ref].reshape( nshape )
  if lowcut :
    dout = rk_fourier.butter_bandpass_filter( dout, lowcut, highcut, 1./dt )
  return dout


def decon_itrace( data, iref, damp=0.002, nchunk=0, ilag0=0, ilag1=0 ) :

  # data is ntrace * nt size
  n1 = data.shape[-1]
  n0 = data.shape[0]

  ref_trace = data[ iref, : ] 

  if nchunk == 0 :
    nchunk = n1

  if ilag1 == 0 :
    ilag1 = n0

  nlag = ilag1 - ilag0 

  cdata = np.zeros( ( n0, nlag ), dtype=np.float )  
 
  ref_data = np.repeat( ref_trace.reshape( (1, n1) ), nchunk, axis=0 )
  print( ref_data.shape, cdata.shape )
  for i0 in range( 0, n0, nchunk ) : 
    i1 = min( i0 + nchunk, n0 )
    iref1 = min( nchunk, i1-i0 )
    cdata[ i0:i1, : ] = decon( ref_data[ :iref1, :], data[ i0:i1, :] 
                              )[ :, ilag0:ilag1]


  return cdata


