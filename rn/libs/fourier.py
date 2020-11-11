#!/usr/bin/env python

import numpy as np
from scipy import signal
import scipy as sp
try :
  import pyfftw
except : 
  print( 'no pyfftw found' )


#=============================================================
# FFT
#=============================================================

def rfft( din, fft_object=None, t=None, dt=None ) :

  try :
    if fft_object == None :
      fft_object = rfft_object(  ) 
      fft_object.set( din ) 
    dout = fft_object.rfft_2d( din ) 
  except :
    dout = np.fft.rfft( din )

  if t is None :
    nt = din.shape[-1]
    #t = np.arange( din.shape[-1] )
    if dt is None :
      dt = 1.
    fout = t2f( nt=nt, dt=dt )
  else :
    fout = t2f( t=t )


  ndout = dout.shape[-1]  

  fout = fout[:ndout]

  if fft_object :
    return dout, fout
  else :
    return dout, fout,  fft_object


def t2f( t=None, dt=None, nt=None ) :
  if type(t) is np.ndarray :
    dt = t[1] - t[0]
    nt = t.shape[0]


  tmax = nt*dt
  fmax = 1./dt
  df = 1./tmax
  f = np.arange( nt ) * df
  return f


class rfft_object( object) :
  def set( self, din ) :
    self.nt  = din.shape[-1]
    self.nyq = self.nt /2
    self.a = pyfftw.empty_aligned( self.nt, dtype='float32')
    self.b = pyfftw.empty_aligned( self.nyq + 1, dtype='complex64')
    self.fft_object = pyfftw.FFTW( self.a, self.b, flags=('FFTW_ESTIMATE',) )
    
  def rfft_2d( self, din ) :
    ntrace = din.shape[0]
    dout = np.zeros( ( ntrace, self.nyq + 1 ), dtype='complex64')
    for itrace in range(ntrace) :
      self.a[:] = din[ itrace, : ]
      self.fft_object()
      dout[ itrace, : ] = self.b
    return dout 
 


def irfft( din ):
    ntrace  = din.shape[0]
    nyq     = din.shape[-1] - 1
    nsample = nyq * 2 
    try :
      a = pyfftw.empty_aligned( nyq + 1, dtype='complex_')
      fft_object = pyfftw.builders.irfft(a, planner_effort = 'FFTW_ESTIMATE' )
      dout = np.zeros( (ntrace, nsample), dtype='float_')
      for itrace in range(ntrace) :
          dout[ itrace, : ] = fft_object( din[ itrace, : ]) 
    except :
      dout = np.fft.irfft( din )
    return dout



#=============================================================
# spectra
#=============================================================

def amp_spectra( d, dt ) :
    
  nt = d.shape[-1]

  f = np.arange( 0, nt/2+1, dtype=np.float )  / dt / float( nt )

  fd = np.abs( np.fft.rfft( d, axis = -1 ) )


    
  return fd, f

def todB( fd ) :
  fd /= np.max(fd)
  fd = 20.*np.log10( fd )
  return fd





def phase_spectra( d, dt ) :
  nt = d.shape[-1]

  f = np.arange( 0, nt/2+1, dtype=np.float )  / dt / float( nt )

  fd = np.angle( np.fft.rfft( d, axis = -1 ) )
    
  return fd, f

#=============================================================
# calculate time and frequency shift
#=============================================================
def time_phase_from_linefit( phz, f, lowf, highf ) :
  i0 = np.where( f > lowf )[0][0]
  i1 = np.where( f < highf )[0][-1]
 
  nf =  len(f) 
  nshape = phz.shape
  ntrace = np.prod( nshape[0:-1] )

  f2   = f[ i0:i1 ]
  nf2  = len(f2)
  nshape2 = np.asarray( phz.shape )

  phz2 = np.unwrap( phz.reshape( ntrace, nf )[ :, i0:i1 ], axis=-1)  
 


  a  = np.polyfit( f2, phz2.T, deg=1 );
  phz0 = a[1].reshape( nshape2[0:-1] )
  t0   = - a[0].reshape( nshape2[0:-1] )  /2. / np.pi
  
  phz0 = np.ma.masked_equal( phz0, 0 )
  t0   = np.ma.masked_equal( t0, 0 )
  phz0_wrap = np.angle( np.exp( 1j* phz0 ) )

  return phz0, t0, phz0_wrap, f2








#=============================================================
# taper
#=============================================================
def hanning_taper( din, ntaper ) :
    fwin  = np.hanning(ntaper*2)
    ftaper = np.concatenate( ( fwin[:ntaper], 
                               np.ones( din.shape[-1] - ntaper *2 ),
                               fwin[ -ntaper: ]
                               ))
    ntrace = din.shape[0]
    dout = np.zeros( din.shape, dtype=din.dtype )
    for itrace in range(ntrace) :
        dout[ itrace, : ] = din[ itrace, : ] * ftaper
    return dout

#=============================================================
# padding
#=============================================================
def padding( din, npad, axis=1) :
  n0, n1 = din.shape

  n1pad = n1 + npad #int( float( n1 ) * pad )

  xpad = np.repeat( np.zeros( ( 1, n1pad - n1 ) ) , n0, axis=0)

  return np.hstack( (din, xpad ) )
 
 

#=============================================================
# resample
#=============================================================
def resample( din, nout ):
    ndim      = din.ndim 
    nin       = din.shape[-1]
    nshape_in = list( din.shape ) 
    if ndim == 1 :
        din = np.reshape( din, ( 1, nin ) )
    elif ndim > 2 :
        ntrace = ndim.size / nin
        din = np.reshape( din, ( ntrace, nin ) )

    ntrace = din.shape[0]
    fdin, f = rfft( din )
    
    nyq_in  = nin/2 
    nyq_out = nout/2
    fdout = np.zeros( ( ntrace, nyq_out + 1 ), np.dtype('complex_') )
    #print nyq_in
    fdout[ :, : nyq_in + 1 ] = fdin[ :, : nyq_in + 1 ]
    dout = irfft( fdout ) * float(nout) / float(nin)
   
    nshape_out = nshape_in; nshape_out[ -1 ] = nout
    dout = np.reshape( dout, nshape_out )
    return dout



#============================================================
# bandpass filter from scipy cookbook
#============================================================

def butter_bandpass( lowcut=None, highcut=None, fs=None, order=5 ) :
  nyq  = 0.5 * fs

  if lowcut and highcut :
    low  = lowcut / nyq
    high = highcut /nyq
    b, a = signal.butter( order, [ low, high ], btype='band') 
  elif not lowcut :
    high = highcut /nyq
    b, a = signal.butter( order, high,  btype='lowpass') 
  else :
    low = lowcut /nyq
    print( 'lowcut filter', low )
    b, a = signal.butter( order, low,  btype='highpass') 
  return b, a

def butter_bandpass_filter( data, lowcut=None, highcut=None, fs=None, 
                            order=5, pad=0.1, causal='n' ) :

  if data.ndim == 1 :
    data = data.reshape( ( 1, data.shape[-1] ) )

  n0, n1 = data.shape
  #print n0, n1

  n1pad = n1 + int( float( n1 ) * pad )

  b, a = butter_bandpass( lowcut, highcut, fs, order=order )

  y = np.zeros( ( n0, n1 ), dtype=np.float )
  xpad = np.zeros( ( n1pad - n1 ) ) 



  if causal == 'y' :
    for i in range( n0 ) :
      y[ i, : ]  = signal.filtfilt( b, a, np.hstack( ( data[ i, :], xpad ) ),
                            axis=-1 )[ : n1 ] 
  else :
    for i in range( n0 ) :
      y[ i, : ]  = signal.filtfilt( b, a, np.hstack( ( data[ i, :], xpad ) ),
                            axis=-1 )[ : n1 ] 

  #print( y )

  return y

def butter_bandpass_filter_inplace( data, lowcut=None, highcut=None, 
                fs=None, order=6 ) :
  b, a = butter_bandpass( lowcut, highcut, fs, order=order )
  for i in range( data.shape[0]) :
    data[ i, : ]  = signal.filtfilt( b, a, data[ i, :] , axis=-1 )



def hanning_bandpass( d, dt,  lowcut=None, highcut=None, ntaper=None,
                      freqs=None ) :
  
  fd, f = rfft( d, dt=dt )

  if freqs is None :
    fd = hanning_bandpass_fd( fd,  df=f[1]-f[0], lowcut=lowcut, highcut=highcut,
                         ntaper=ntaper ) 

  else :
    fd = hanning_bandpass_fd_freqs( fd, df=f[1]-f[0], freqs=freqs ) 

  return irfft( fd )



def hanning_bandpass_fd( fd, df, lowcut=None, highcut=None, ntaper=None ) :
  if lowcut :
    nlowcut = int( lowcut/df )
  if highcut :
    nhighcut = int( highcut/df )
  ftaper = np.hanning( ntaper * 2 )
  nf =fd.shape[-1]
  if lowcut and highcut :
    print( 'here' )
    f = np.concatenate( ( np.zeros( nlowcut-ntaper, dtype=fd.dtype ),
                      ftaper[ :ntaper ],
                      np.ones( -nlowcut+nhighcut, dtype=fd.dtype ),
                      ftaper[ ntaper: ],
                      np.zeros( nf - nhighcut - ntaper, dtype=fd.dtype ) ) )
    print( f.shape )
  elif not lowcut :
    f = np.concatenate( ( np.ones( nhighcut, dtype=fd.dtype ),
                      ftaper[ ntaper: ],
                      np.zeros( nf - nhighcut-ntaper, dtype=fd.dtype ) ) )

  else :
    print( nf, nlowcut, ntaper, nf-nlowcut, ntaper )
    print( np.ones( nf-nlowcut-ntaper ) )

    f = np.concatenate( ( np.zeros( nlowcut, dtype=fd.dtype ),
                      ftaper[ :ntaper ],
                      np.ones( nf-nlowcut-ntaper, dtype=fd.dtype )) )

  fdout = np.zeros_like( fd )
  for i in range( fd.shape[0] ) :
    fdout[ i, : ] = fd[ i, : ] * f

  return fdout


def hanning_bandpass_fd_freqs( fd, df, freqs) :
  nfreqs = ( freqs / df ).astype( dtype=np.int )
  nhighcut = nfreqs[2]
  nlowcut  = nfreqs[1]
  ntaper1  = nfreqs[1] - nfreqs[0]
  ntaper2  = nfreqs[3] - nfreqs[2]

  ftaper1 = np.hanning( ntaper1 * 2 )
  ftaper2 = np.hanning( ntaper2 * 2 )
  nf =fd.shape[-1]

  if nlowcut and nhighcut :
    f = np.concatenate( ( np.zeros( nlowcut-ntaper1, dtype=fd.dtype ),
                      ftaper1[ :ntaper1 ],
                      np.ones( -nlowcut+nhighcut, dtype=fd.dtype ),
                      ftaper2[ ntaper2: ],
                      np.zeros( nf - nhighcut - ntaper2, dtype=fd.dtype ) ) )
    print( f.shape )
  elif not lowcut :
    f = np.concatenate( ( np.ones( nhighcut, dtype=fd.dtype ),
                      ftaper[ ntaper: ],
                      np.zeros( nf - nhighcut-ntaper, dtype=fd.dtype ) ) )

  else :
    print( nf, nlowcut, ntaper, nf-nlowcut, ntaper )
    print( np.ones( nf-nlowcut-ntaper ) )

    f = np.concatenate( ( np.zeros( nlowcut, dtype=fd.dtype ),
                      ftaper[ :ntaper ],
                      np.ones( nf-nlowcut-ntaper, dtype=fd.dtype )) )

  fdout = np.zeros_like( fd )
  for i in range( fd.shape[0] ) :
    fdout[ i, : ] = fd[ i, : ] * f

  return fdout

