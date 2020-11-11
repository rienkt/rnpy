#!/usr/bin/env python
"""

  Mask

  2d mask-related modules here 
  This is nearly same as those in fk.py


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import matplotlib.path   as mplpath
import rn.libs.filters   as rn_filters

#======================================================================
# classes / functions
#======================================================================

def rn_mask( polyx, polyy, x, y ) :
  return mask( polyx, polyy, x, y ) 

def mask( polyx, polyy, x, y ) :
  # originally prepared for fk filtering
  nx = len( x )
  ny = len( y )

  poly_verts = [ ( polyx[0], polyy[0] ) ]
  for i in range( len(polyx)-1, -1, -1 ) :
    poly_verts.append( (polyx[i], polyy[i] ) )

  xx, yy = np.meshgrid( x, y )
  points = np.vstack( ( xx.flatten(), yy.flatten() ) ).T 

  polypath = mplpath.Path( poly_verts )
  
  print xx.shape

#  grid = polypath.contains_points( points ).reshape( (ny, nx) )
  grid = polypath.contains_points( points ).reshape( (ny, nx) ).T

  mask = np.ones( grid.shape, dtype=np.float )
  mask[ np.where( grid ) ] = 0.


  return mask



def mask_smooth( mask, xsigma, ysigma ) :
  mask = rn_filters.gaussian( mask, ysigma, xsigma )
  return mask


def mask_hanning( x, z, ztop, ztaper, nztops=None) :
  nx = len(x)
  nz = len(z)
  dz = z[1] - z[0]
  nztaper = np.int( ztaper/dz )
  mask = np.zeros( ( nx, nz ), dtype=np.float )
  if type(z) is float :
    nztop = np.argmin( np.abs( z-ztop) )

    mask[ :, nztop:(nztop+nztaper) ] = np.tile( 
                            np.hanning( nztaper*2 )[:nztaper] , 
                                      (nx,1))
    mask[ :, (nztop+nztaper): ] = 1.
  else :
    for ix in range( nx ) :
      if type(nztops) is np.ndarray  :
        nztop = nztops[ix] 
      else :
        nztop = np.argmin( np.abs( z - ztop[ix] ) )
      nztaper = np.int( ztaper/dz )

      mask[ ix, nztop:(nztop+nztaper) ] = np.hanning( nztaper*2 )[:nztaper] 
      mask[ ix, (nztop+nztaper): ] = 1.

  return mask


