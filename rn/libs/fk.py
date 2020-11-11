#!/usr/bin/env python
"""

  fk.py

  Analyze and filter fk domain


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
import jdoepfert.roipoly as roipoly
import rn.libs.filters as rn_filters

#======================================================================
# classes / functions
#======================================================================


def rn_fk( din, dt, dx ) :
  nx, nt = din.shape
  df = 1./nt/dt
  dk = 1./nx/dx

  fd = np.fft.fftshift( np.fft.rfft2( din ), 0 )

  f = np.arange( 0, nt/2 +1 , dtype=np.float ) * df;
  k = np.arange( 0, nx , dtype=np.float ) * dk - nx/2*dk;
  
  return fd, f, k

def rn_ifk( fd ) :
  d = np.fft.irfft2( np.fft.ifftshift( fd, 0 ) )
  return d

def rn_fk_plot( din, f, k, ax ) :
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

  mask = np.ones( grid.shape, dtype=np.float )
  mask[ np.where( grid ) ] = 0.


  return mask


def rn_mask_smooth( mask, ksigma, fsigma ) :
  mask = rn_filters.gaussian( mask, ksigma, fsigma )
  return mask

#def rn_apply_fk( fdin, mask ) :
#  fd = fdin * mask
#  d = np.fft.irfft2( np.fft.ifftshift( fd, 0 ) )
#  return d
  
  
def rn_reverse_k( f, k ) :
  fout = np.concatenate( (f, f) )
  kout = np.concatenate( (k,-k) )
  return fout, kout

def rn_read_poly( fname ) :
  with open( fname ) as f :
    lines = f.readlines() 

  f = np.array( [ line.split()[0] for line in lines ], 
                        dtype=np.float ) 
  k = np.array( [ line.split()[1] for line in lines ], 
                        dtype=np.float ) 
  return f, k
