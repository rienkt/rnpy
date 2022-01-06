#!/usr/bin/env python
"""

  kk.py

  Analyze and filter kk domain


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path   as mplpath
import rn.libs.filters as rn_filters

#======================================================================
# classes / functions
#======================================================================


def kk( din, dz, dx ) :
  nx, nz = din.shape
  dkz = 1./nz/dz
  dkx = 1./nx/dx

  #fd = np.fft.fftshift( np.fft.rfft2( din ), 0 )
  fd = np.fft.fftshift( np.fft.fft2( din ), -1 )
  fd = np.fft.fftshift( fd, 0 )
  #fd =  np.fft.rfft( din )

  #kz = np.arange( 0, nz/2 +1 , dtype=np.float ) * dkz;
  kz = np.arange( 0, nz , dtype=np.float ) * dkz - nz/2*dkz;
  kx = np.arange( 0, nx , dtype=np.float ) * dkx - nx/2*dkx;
  
  return fd, kx, kz

def ikk( fd ) :
  fd = np.fft.ifftshift( fd, 0 )
  fd = np.fft.ifftshift( fd, -1 )
  d = np.fft.ifft2( fd )
  return d

def rn_kk_plot( din, f, k, ax ) :
  extent = ( k[0], k[-1], f[0], f[-1] )
  ax.imshow( np.flipud( np.abs( din ).transpose() ) / np.max( np.abs( din ) ), 
                extent=extent, aspect='auto', cmap='magma',
                vmin=0, vmax=0.1 )

def rn_plot_v( k, v, ax ) :
  ax.plot( k, k*v, 'w' )
  ax.plot( k, -k*v,'w' )


# this function doesn't worn
def rn_pick_poly( ax, fig ) :
  poly = roipoly.roipoly( roicolor = 'w', ax=ax, fig=fig )
  polyf = poly.allypoints
  polyk = poly.allxpoints
  return polyf, polyk


def rn_mask( polyf, polyk, f, k ) :
  nf = len( f )
  nk = len( k )

  poly_verts = [ ( polyk[0], polyf[0] ) ]
  for i in range( len(polyf)-1, -1, -1 ) :
    poly_verts.append( (polyk[i], polyf[i] ) )

  ff, kk = np.meshgrid( f, k )
  points = np.vstack( ( kk.flatten(), ff.flatten() ) ).T 

  polypath = mplpath.Path( poly_verts )

  grid = polypath.contains_points( points ).reshape( (nk, nf) )

  mask = np.ones( grid.shape, dzype=np.float )
  mask[ np.where( grid ) ] = 0.


  return mask


def rn_mask_smooth( mask, ksigma, fsigma ) :
  mask = rn_filters.gaussian( mask, ksigma, fsigma )
  return mask

#def rn_apply_kk( fdin, mask ) :
#  fd = fdin * mask
#  d = np.fft.irfft2( np.fft.ifftshift( fd, 0 ) )
#  return d
  
  
def rn_reverse_k( f, k ) :
  fout = np.concatenate( (f, f) )
  kout = np.concatenate( (k,-k) )
  return fout, kout
#!/usr/bin/env python
"""

  FILTERS.py
  
  various filters to be stored here?

  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
from scipy import ndimage
import copy

#======================================================================
# classes / functions
#======================================================================

# hanning filter

def hanning_taper( din, ntaper ) :
    fwin  = np.hanning(ntaper*2)
    ftaper = np.concatenate( ( fwin[:ntaper], 
                               np.ones( din.shape[-1] - ntaper *2 ),
                               fwin[ -ntaper: ]
                               ))
    ntrace = din.shape[0]
    dout = np.zeros( din.shape, dtype='float_' )
    for itrace in range(ntrace) :
        dout[ itrace, : ] = din[ itrace, : ] * ftaper
    return dout



# 
def dim_convert_to2D( din ) :
  ndim      = din.ndim 
  nin       = din.shape[-1]
  nshape_in = list( din.shape ) 
  if ndim == 1 :
    din = np.reshape( din, ( 1, nin ) )
  elif ndim > 2 :
    ntrace = din.size / nin
    din = np.reshape( din, ( ntrace, nin ) )
  
  return din, nshape_in

# time-domain filters
def remove_dc( din, i0=0, i1=-1 ) :
  ndim      = din.ndim 
  nin       = din.shape[-1]
  nshape_in = list( din.shape ) 
  if ndim == 1 :
    din = np.reshape( din, ( 1, nin ) )
  elif ndim > 2 :
    ntrace = din.size / nin
    din = np.reshape( din, ( ntrace, nin ) )

  n0, n1 = din.shape
  dmean = np.mean( din[ :, i0:i1] , axis=-1 )

  for i0 in range( n0 ) :
    din[ i0, : ] -= dmean[ i0 ]

  din = np.reshape( din, nshape_in )
  print( nshape_in, din.shape )


  return din, dmean

def time_window_gauss( din, t, ot=0., tsigma=1.0 ) :

  din, nshape_in = dim_convert_to2D( din )
  n0, n1 = din.shape

  fgauss = np.repeat( np.exp( -( t-ot )**2 / tsigma**2 *0.5 ), n0
                    ).reshape( n1, n0 ).T

  dout = din * fgauss
  return np.reshape( dout, nshape_in ), fgauss
  
# window data 
#                 tbeg
# 0 .... tw0 .... tw1 ............ tw2 ... tw3 .... T
# 0       0 taper   window_length      taper 0       0
# 
# careful! window starts from fbreak + top_tshift - top_taper
#def rk_time_window( din, top_tshift, top_taper, window_length, bottom_taper ) :
def time_window_hanning( din, t, tbeg, tend, top_taper, bottom_taper ) :
  if din.ndim == 1 :
    d = din.reshape( ( 1, din.shape[-1] ) )
  else :
    d = copy.copy( din ) 
  dt = t[1] - t[0]
  nt = d.shape[-1]

  top_ntaper    = int( top_taper /dt )
  bottom_ntaper = int( bottom_taper / dt )

  ntbeg = max( np.argmin( np.abs( t-tbeg ) ),  0 )
  ntend = min( np.argmin( np.abs( t-tend ) ), nt )

  top_ntaper = min( top_ntaper, ntbeg )
  bottom_ntaper = min( bottom_ntaper, nt-ntend )

  ntime_window = ntend - ntbeg

  top_filter    = np.hanning( top_ntaper*2 )[ : top_ntaper ]
  bottom_filter = np.hanning( bottom_ntaper*2 )[ bottom_ntaper : ]
  time_window_filter = np.hstack( ( top_filter, 
                                    np.ones( ntime_window ),
                                    bottom_filter ) )
  ntime_window = len( time_window_filter )


  itime_window_0 = ntbeg - top_ntaper
  itime_window_1 = itime_window_0 + ntime_window

  f = np.zeros( nt, dtype=np.float )
  f[ itime_window_0 : itime_window_1 ] = time_window_filter


#  d[ :, : itime_window_0 ]                *= 0.
#  d[ :, itime_window_0 : itime_window_1 ] *= time_window_filter 
#  d[:, itime_window_1 : ]                *= 0.
  d *= f

  if din.ndim == 1 :
    d = d.squeeze()
  return d, f

  


def time_window( din, t, fbreak, 
                top_tshift, top_taper, window_length, bottom_taper,
                    inplace=1 ) :
  # do not pass class. only pass ndarrays
  
  if inplace == 0 :
    d = np.zeros_like( din )
  else :
    d = din

  dt = t[1]-t[0] 
  nt = t.size

  top_ntshift   = int( top_tshift /dt )  
  top_ntaper    = int( top_taper /dt )
  bottom_ntaper = int( bottom_taper / dt )
  ntime_window  = min( int( window_length / dt ), nt - top_ntshift - bottom_ntaper )


  top_filter    = np.hanning( top_ntaper*2 )[ : top_ntaper ]
  bottom_filter = np.hanning( bottom_ntaper*2 )[ bottom_ntaper : ]
  time_window_filter = np.hstack( ( top_filter, 
                                    np.ones( ntime_window ),
                                    bottom_filter ) )
  ntime_window = len( time_window_filter )

  ntrace = d.shape[ 0 ] 
  # make arrays longer to avoid the end of window is too long...
  dtmp = np.concatenate( ( d, np.zeros( ( ntrace, ntime_window ), dtype=np.float ) ), axis=-1 )

  for itrace in range( ntrace ) :
    if fbreak.mask[ itrace ] :
      dtmp[ itrace, : ] *= 0.
    else :
      #nfbreak = np.argmin( abs( t - fbreak.times[ itrace ] ) )
      nfbreak = np.argmin( abs( t - fbreak[ itrace ] ) )

      itime_window_0 = max( nfbreak + top_ntshift - top_ntaper, 0 )
      itime_window_1 = itime_window_0 + ntime_window
      #print( itrace, itime_window_1, itime_window_0, ntime_window )
      dtmp[ itrace, : itime_window_0 ]                *= 0.
      dtmp[ itrace, itime_window_0 : itime_window_1 ] *= time_window_filter 
      dtmp[ itrace, itime_window_1 : ]                *= 0.


  return dtmp[ :, :nt ]



# spatial filters
def moving( data, nsize, mode='nearest' ) :
  return ndimage.filters.uniform_filter( data, size = nsize, mode = mode )

def gaussian( data, fsigma1, fsigma2 ) :
  return ndimage.filters.gaussian_filter( data, ( fsigma1, fsigma2 ) )

def gaussian_mask( data, fsigma1, fsigma2 ) :
  n0, n1 = data.shape
  dout = copy.copy( data )
  for i0 in range(n0) :
    idx = np.where( data.mask[ i0, : ] == False )
    dout[ i0, idx ] = ndimage.filters.gaussian_filter1d( 
                        data[ i0, idx ], fsigma2, mode='nearest' )

  dout = dout.T
  for i1 in range(n1) :
    idx = np.where( dout.mask[ i1, : ] == False )
    dout[ i1, idx ] = ndimage.filters.gaussian_filter1d( 
                        dout[ i1, idx ], fsigma1, mode='nearest' )

  return dout.T

 
def gaussian1d( data, fsigma1 ) :

  if data.ndim == 1 :
    dout = ndimage.filters.gaussian_filter1d( data, fsigma1 )
  else :

    n0 = data.shape
    dout = copy.copy( data )
    for i0 in range(n0) :
      dout[ i0, : ] = ndimage.filters.gaussian_filter1d( 
                          data[ i0, : ], fsigma1 )
  return dout
 
 
def gaussian1d_mask( data, fsigma1 ) :
  n0, n1 = data.shape
  dout = copy.copy( data )
  for i0 in range(n0) :
    idx = np.where( data.mask[ i0, : ] == False )
    dout[ i0, idx ] = ndimage.filters.gaussian_filter1d( 
                        data[ i0, idx ], fsigma1 )
 
  return dout 
 

def median( d, size, mask=None )  : 
  
  import skimage.filters.rank 
  import skimage

  try :
    mask = np.logical_not( d.mask ) 
  except :
    print( 'not masked array' )


  dim = skimage.img_as_ubyte( d/np.max(np.abs(d)))
  selem = np.ones( size )

  #print selem

  doutim = skimage.filters.rank.median( dim, selem = selem, mask=mask )

  dout = skimage.img_as_float( doutim )  * np.max( np.abs(d) )


  return dout



def rn_read_poly( fname ) :
  with open( fname ) as f :
    lines = f.readlines() 

  f = np.array( [ line.split()[0] for line in lines ], 
                        dzype=np.float ) 
  k = np.array( [ line.split()[1] for line in lines ], 
                        dzype=np.float ) 
  return f, k
