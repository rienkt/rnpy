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
import rk.libs.filters as rk_filters

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

def rk_kk_plot( din, f, k, ax ) :
  extent = ( k[0], k[-1], f[0], f[-1] )
  ax.imshow( np.flipud( np.abs( din ).transpose() ) / np.max( np.abs( din ) ), 
                extent=extent, aspect='auto', cmap='magma',
                vmin=0, vmax=0.1 )

def rk_plot_v( k, v, ax ) :
  ax.plot( k, k*v, 'w' )
  ax.plot( k, -k*v,'w' )


# this function doesn't work
def rk_pick_poly( ax, fig ) :
  poly = roipoly.roipoly( roicolor = 'w', ax=ax, fig=fig )
  polyf = poly.allypoints
  polyk = poly.allxpoints
  return polyf, polyk


def rk_mask( polyf, polyk, f, k ) :
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


def rk_mask_smooth( mask, ksigma, fsigma ) :
  mask = rk_filters.gaussian( mask, ksigma, fsigma )
  return mask

#def rk_apply_kk( fdin, mask ) :
#  fd = fdin * mask
#  d = np.fft.irfft2( np.fft.ifftshift( fd, 0 ) )
#  return d
  
  
def rk_reverse_k( f, k ) :
  fout = np.concatenate( (f, f) )
  kout = np.concatenate( (k,-k) )
  return fout, kout

def rk_read_poly( fname ) :
  with open( fname ) as f :
    lines = f.readlines() 

  f = np.array( [ line.split()[0] for line in lines ], 
                        dzype=np.float ) 
  k = np.array( [ line.split()[1] for line in lines ], 
                        dzype=np.float ) 
  return f, k
