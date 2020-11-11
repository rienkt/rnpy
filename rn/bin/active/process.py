#!/usr/bin/env python
from __future__ import print_function



"""

  PROCESS.py

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np

from .core import rk_binary
import rk.libs as rk_lib
import rk.libs.fourier as rk_f
import rk.libs.filters as rk_filters
import rk.libs.normalize as rk_normalize
import rk.libs.lmo 
import copy

#======================================================================
# classes / functions
#======================================================================

# calculate amplitude spectra


def rk_amp_spectra( d ) :
  d.fdata, d.f = rk_f.amp_spectra( d.data, d.dt )
  return d 

def rk_phase_spectra( d ) :
  d.fphz, d.f = rk_f.phase_spectra( d.data, d.dt )
  return d

# calculate lmo
def rk_lmo( d, tshift, inplace=0, otshift=0. ) :
  if inplace == 0 :
    dout = copy.copy(d) 

  dout.ot = otshift

  if d.gather == 'all'  :
    dout.data, dout.t = rk.libs.lmo.lmo( 
                                 d.data.reshape( d.srcs.n*d.rcvs.n, d.nt ),
                                 d.dt, tshift.reshape( d.srcs.n*d.rcvs.n ),
                                 otshift=otshift)
    dout.data = dout.data.reshape( ( d.srcs.n, d.rcvs.n, d.nt ) )
  else:
    dout.data, dout.t = rk.libs.lmo.lmo( d.data, d.dt, tshift, otshift=otshift )
  


  return dout


# calculate reduced time
#def lmo( d, dt, tshift ) :
#  nt = d.shape[-1] 
#  ntrace = d.shape[0]
#  fd = np.fft.rfft( d, axis=-1 )
#  f = np.arange( 0, nt/2+1, dtype=np.float ) / dt / float(nt) 
#  nf = f.shape[-1]
#  ttshift = np.repeat( tshift.reshape( ntrace, 1 ), nf, axis=-1 )
#
#  fd *= np.exp( 2j * np.pi * f * ttshift )
#
#  dout = np.fft.irfft( fd, axis=-1 )
#  return dout

def time_phase_shift( d, dt, tshift, phzshift ) :
  nt = d.shape[-1] 
  ntrace = d.shape[0]
  fd = np.fft.rfft( d, axis=-1 )
  f = np.arange( 0, nt/2+1, dtype=np.float ) / dt / float(nt) 
  nf = f.shape[-1]
  ttshift = np.repeat( tshift.reshape( ntrace, 1 ), nf, axis=-1 )
  pphzshift = np.repeat( phzshift.reshape( ntrace, 1 ), nf, axis=-1 )

  fd *= np.exp( -2j * np.pi * f * ttshift + 1j*pphzshift)

  dout = np.fft.irfft( fd, axis=-1 )
  return dout

 


def normalize( d, axis=-1 ) :
  d.data = rk_normalize.normalize( d.data,  axis=axis )
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
  return d


# bandpass filter for all functions
def rk_bandpass( din, lowcut, highcut, ntaper, pad=1.0, order=3  ) :

  d = copy.copy( din )
  if din.gather == 'all' :
    d.data = d.data.reshape( d.srcs.n * d.rcvs.n , d.nt ) 



  dtaper = rk_f.rk_hanning_taper( d.data, ntaper ) 
  d.data = rk_f.butter_bandpass_filter( dtaper, lowcut, highcut,1./din.dt, pad=pad,
                                        order=order )[ :, :d.nt ]

  if din.gather == 'all' :
    d.data = d.data.reshape( d.srcs.n, d.rcvs.n, d.nt ) 

  return d
 


# window data 
#             fbreak + top_tshift
# 0 .... tw0 .... tw1 ............ tw2 ... tw3 .... T
# 0       0 taper   window_length      taper 0       0
# 
# careful! window starts from fbreak + top_tshift - top_taper

def rk_time_window( din, top_tshift, top_taper, window_length, bottom_taper,
                    inplace=1 ) :
  
  #d = copy.copy(din)
  if inplace == 0 :
    d = rk_binary( ref=din )
  else :
    d = din


  top_ntshift   = int( top_tshift /d.dt )  
  top_ntaper    = int( top_taper /d.dt )
  bottom_ntaper = int( bottom_taper / d.dt )
  ntime_window  = min( int( window_length / d.dt ), din.nt - top_ntshift - bottom_ntaper )


  top_filter    = np.hanning( top_ntaper*2 )[ : top_ntaper ]
  bottom_filter = np.hanning( bottom_ntaper*2 )[ bottom_ntaper : ]
  #print top_filter.shape, bottom_filter.shape, np.zeros( ntime_window).shape
  time_window_filter = np.hstack( ( top_filter, 
                                    np.ones( ntime_window ),
                                    bottom_filter ) )
  ntime_window = len( time_window_filter )
  #print time_window_filter

  d.data = copy.copy( din.data )


  if d.gather == 'all' :
    d.fbreak.times = d.fbreak.times.reshape( d.srcs.n * d.rcvs.n )
    d.data = d.data.reshape( d.srcs.n * d.rcvs.n, d.nt ) 


  if type( d.fbreak.times.mask ) is np.bool_ :
    d.fbreak.times.mask = np.zeros( d.srcs.n * d.rcvs.n, dtype=np.bool )

  ntraces = d.data.shape[ 0 ] 
  for itrace in range( ntraces ) :
    if d.fbreak.times.mask[ itrace ] :
      d.data[ itrace, : ] *= 0.
    else :
      nfbreak = np.argmin( abs( d.t - d.fbreak.times[ itrace ] ) )
  #    print( nfbreak )

      itime_window_0 = max( nfbreak + top_ntshift - top_ntaper, 0 )
      itime_window_1 = itime_window_0 + ntime_window
      #print( time_window_filter.shape, itime_window_0, itime_window_1 )
      #print nfbreak, itime_window_0, itime_window_1
      d.data[ itrace, : itime_window_0 ]                *= 0.
      d.data[ itrace, itime_window_0 : itime_window_1 ] *= time_window_filter 
      d.data[ itrace, itime_window_1 : ]                *= 0.

  if d.gather == 'all' :
    d.fbreak.times = d.fbreak.times.reshape( ( d.srcs.n, d.rcvs.n ) )
    d.data = d.data.reshape( ( d.srcs.n, d.rcvs.n, d.nt ) )

   # print d.data[ itrace, : ] 

 # print din.data
 # print d.data

  return d


def rk_time_window_v0_v1( din,  v0in, v1in, top_taper, bottom_taper, 
                          top_tshift=0., bottom_tshift=0., inplace=1 ) :
  if inplace == 0 :
    d = rk_binary( ref=din )
    d.data = copy.copy( din.data )
  else :
    d = din

  v0 = min( v0in, v1in )
  v1 = max( v0in, v1in )


  d.set_cmp() # get offse
  t0 = d.offset / v1 + top_tshift
  t1 = d.offset / v0 + bottom_tshift

  d.data, nshape_in = rk_filters.dim_convert_to2D( d.data )

  n0, n1 = d.data.shape

  t0 = t0.reshape( n0 )
  t1 = t1.reshape( n0 )

  for i0 in range( n0 ) :
    if t0[i0] <   t1[i0] :
      d.data[i0,:] = rk_filters.time_window_hanning( 
                        d.data[ i0, : ], d.t, t0[i0], t1[i0],
                        bottom_taper, top_taper )
    else :
      d.data[i0,:] *= 0.

  d.data = d.data.reshape( nshape_in ) 
  return d
 


  

 


def rk_top_mute( din, window_length, top_taper, inplace=1 ) :
  fbreak_sav = din.fbreak.times
  din.fbreak.times = np.ma.zeros( fbreak_sav.shape, dtype=np.float )            
  din.fbreak.times.mask = np.zeros( fbreak_sav.shape, dtype=bool )
  dout = rk_time_window( din, window_length+top_taper, top_taper, din.nt*din.dt, 0, inplace=inplace )
  dout.fbreak.times = fbreak_sav
  return dout 

def rk_bottom_mute( din, window_length, bottom_taper ) :
  fbreak_sav = din.fbreak.times
  din.fbreak.times = np.ma.zeros( fbreak_sav.shape, dtype=np.float )            
  din.fbreak.times.mask = np.zeros( fbreak_sav.shape, dtype=bool )
  dout = rk_time_window( din, 0, 0, din.dt*din.nt-window_length, bottom_taper )
  dout.fbreak.times = fbreak_sav
  return dout 








  


