#!/usr/bin/env python
"""
  processing 

  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import copy
import oshima.lib as rn_lib
from .core import rn_binary
#======================================================================
# Parameters
#======================================================================



#======================================================================
# functions 
#======================================================================

#------------------------------------------------------------------------
# extend rn_binary classes to include simple processes
#------------------------------------------------------------------------
class rn_binary_w_process( rn_binary ) :
  def remove_dc( self ) :

   # remove DC & normalize
    self.mean = np.mean( self.data, axis = -1 )

    for ichannel in range( self.nchannels ) :
      self.data[ ichannel, : ] = self.data[ ichannel, : ] - self.mean[ ichannel ]


  def normalize( self ) :
    rn_normalize.normalize( self.data )

  def bandpass( self, lowcut, highcut, order=5 ) :
    rn_f.butter_bandpass_filter_inplace( 
                             self.data, lowcut, highcut, 
                            float(self.samp_freq), order=order)

  def rfft( self ): 
#    self.data  = rn_lib.rn_hanning_taper( self.data, 10 )

    if self.fft_object is None :
      self.fft_object = rn_lib.rn_rfft_object()
      self.fft_object.set( self.data )

    print self.nsamples, self.data.shape
    
    self.fdata = self.fft_object.rfft_2d( self.data )


    self.f     = np.arange( 0., self.nsamples/2 + 1, dtype=np.float 
                          )  / float( self.nsamples ) * self.samp_freq


#------------------------------------------------------------------------
# extend rn_binary classes to include simple processes
#------------------------------------------------------------------------
def copy_header( d ) :
  dout = rn_binary() 
  
  dout.nchannels = d.nchannels
  dout.channels  = d.channels
  dout.t0        = d.t0 
  dout.t1        = d.t1
  dout.totalsec  = d.totalsec
  dout.nsamples  = d.nsamples
  dout.samp_freq = d.samp_freq

  dout.fft_object = d.fft_object

  return dout

def extract_time( d, t0, t1, iskip=1 ) :
  d.set_t()
  
  dout = copy_header( d ) 
  dout.samp_freq = d.samp_freq / float( iskip )
  

  if t0 > d.t0 :
    it0 = d.t.index( t0 ) 
    dout.t0 = d.t[ it0 ];
  else :
    dout.t0 = d.t0
    it0 = 0
 
  if t1 < d.t1 :
    it1 = d.t.index( t1 ) 
    it1 = int( ( ( t1 - d.t0 ).total_seconds() +1 )* d.samp_freq )
    dout.t1 = d.t[it1]
    it1 = it1 + 1
  else :
    dout.t1 = d.t1
    it1 = d.nsamples - it0
 
 
  dout.set_totalsec_nsamples()
  dout.data = d.data.transpose()[ it0 : it1 : iskip, : ].transpose()
  dout.set_t()

  return dout

def extract_channel( d, chlist  ) :

  dout = copy_header( d ) 

  dout.channels  = chlist 
  cout.nchannels = len(chlist)
  dout.initialize( 0. )

  for idx, ch in enumerate( chlist ) : 
    ich = d.channels.index( ch )
    dout.data[ idx, : ] = d.data[ ich, : ].reshape( ( 1, d.nsamples) )


  return dout
  
 


def normalize( din) :
  dout = copy.deepcopy( din )
  dout.data = rn_lib.normalize( din.data )
  return dout

def bandpass( din, lowcut, highcut, order=5 ) :
  dout = copy.deepcopy( din )
  dout.data = rn_lib.butter_bandpass_filter( 
                              din.data, lowcut, highcut, 
                              float(din.samp_freq), order=order)
  return dout
