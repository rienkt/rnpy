#!/usr/bin/env python
from __future__ import print_function
"""

  checkerboard


  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================


def checkerboard( nx, nz, nx_checker, nz_checker ) :

  # first create slightly larger checke3rboard patterns
  nchecker_x = nx / nx_checker / 2 
  nchecker_z = nz / nz_checker / 2

  print( nchecker_x, nchecker_z )

  if nchecker_x * 2 <  nx :
    nchecker_x += 1 
    print( nchecker_x )
    ix0 = ( nchecker_x *nx_checker * 2- nx ) / 2 
  else :
    ix0 = 0 
 
  if nchecker_z *2 <  nz :
    nchecker_z += 1
    iz0 = ( nchecker_z * nz_checker * 2- nz ) / 2
  else :
    iz0 = 0


  a = np.ones( ( nx_checker, nz_checker), dtype=np.float32 )
  b = -np.ones( ( nx_checker, nz_checker ), dtype=np.float32 )

  c = np.row_stack( nchecker_x * ( a, b ) )
  d = np.row_stack( nchecker_x * ( b, a ) )

  e = np.column_stack( nchecker_z * ( c, d ))
  
  print(  ix0, iz0 )
 
  return e[ix0:(ix0+nx), iz0:(iz0+nz)]


def checkerboard_sin( nx, nz, nx_checker, nz_checker ) : 
  # nx_checker and nz_checker doesn't have to be innteger



  # first create slightly larger checkeerboard patterns
  nchecker_x = int( nx / nx_checker )
  nchecker_z = int( nz / nz_checker )

  px = nx_checker * 2.
  pz = nz_checker * 2.

  print( nchecker_x, nchecker_z )

  if nchecker_x  <  nx :
    x0 = + ( nx - nchecker_x *nx_checker )  / 2. - nchecker_x / 2.
    nchecker_x += 2
  else :
    x0 = nchecker_x / 2. 
 
  if nchecker_z <  nz :
    z0 = + ( nz - nchecker_z *nz_checker )  / 2. - nchecker_z / 2.
    nchecker_z += 2
  else :
    z0 = nchecker_z / 2.

  # define meshgrid
  x = np.arange( nx, dtype=np.float )
  z = np.arange( nz, dtype=np.float )
  zz, xx = np.meshgrid( z, x )

  m = np.sin( ( zz - z0 ) / pz * 2* np.pi ) * np.sin( ( xx - x0 )/px * 2 *np.pi )


  return  m

def checkerboard_gauss( nx, nz, nx_checker, nz_checker ) : 
  # nx_checker and nz_checker doesn't have to be innteger



  # first create slightly larger checkeerboard patterns
  nchecker_x = int( nx / nx_checker )
  nchecker_z = int( nz / nz_checker )
  gx_checker = nx_checker / np.exp(1)
  gz_checker = nz_checker / np.exp(1) 

  print( nchecker_x, nchecker_z )

  if nchecker_x  <  nx :
    x0 = + ( nx - nchecker_x *nx_checker )  / 2. - nchecker_x / 2.
    nchecker_x += 2
  else :
    x0 = nchecker_x / 2. 
 
  if nchecker_z <  nz :
    z0 = + ( nz - nchecker_z *nz_checker )  / 2. - nchecker_z / 2.
    nchecker_z += 2
  else :
    z0 = nchecker_z / 2.

  # define meshgrid
  x = np.arange( nx, dtype=np.float )
  z = np.arange( nz, dtype=np.float )
  zz, xx = np.meshgrid( z, x )

  xcheckers = np.arange( nchecker_x, dtype=np.float ) * nx_checker + x0
  zcheckers = np.arange( nchecker_z, dtype=np.float ) * nz_checker + z0
  print( xcheckers )

  a = np.ones( ( 1, 1), dtype=np.float32 )
  b = -np.ones( ( 1, 1 ), dtype=np.float32 )
  c = np.row_stack( nchecker_x * ( a, b ) )
  d = np.row_stack( nchecker_x * ( b, a ) )
  e = np.column_stack( nchecker_z * ( c, d ))
  
  m = np.zeros( ( nx, nz ), dtype=np.float )
 
  for ix in range(nchecker_x) :
    xchecker = xcheckers[ ix ]
    for iz in range( nchecker_z ) :
      zchecker = zcheckers[ iz ]
      m += np.exp( - ( ( xx - xchecker ) / gx_checker ) **2 -
                   ( ( zz - zchecker ) / gz_checker ) **2 ) * e[ ix, iz ]



  return  m

