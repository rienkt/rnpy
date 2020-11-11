#!/usr/bin/env python
"""

  PLOT.py

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import matplotlib.pyplot as plt

from .core import rk_binary

from rk.plot.format import AxesFormat

#======================================================================
# classes / functions
#======================================================================


def image_gather_cw( ax, d, axpar, dflag='md', fbflag=1 ) :
  if d.gather is 'shot' :
    image_shot_gather_cw( ax, d, axpar, dflag=dflag, fbflag=fbflag ) 
  elif d.gather is 'receiver' :
    image_receiver_gather_cw( ax, d, axpar, dflag=dflag, fbflag=fbflag ) 
  elif d.gather is 'offset' :
    image_offset_gather_cw( ax, d, axpar, fbflag=fbflag ) 
  elif d.gather is 'cmp' :
    image_cmp_gather_cw( ax, d, axpar, fbflag=fbflag ) 

#  else d.gather is 'samelevel' :


def image_shot_gather_cw( ax, d, axpar, dflag='md', fbflag=1 ) :
  if axpar.vmin is None :
    axpar.vmin = -0.8 * np.max( np.abs( d.data ) )

  if axpar.vmax is None :
    axpar.vmax = 0.8 * np.max( np.abs( d.data ) )

  if dflag == 'md' :
    rz = d.rcvs.id.astype( np.float ) 
  else :
    rz = d.rcvs.z

  ax.imshow( d.data, extent = ( d.t[0], d.t[-1], rz[-1], rz[0] ),
           vmin = axpar.vmin, vmax = axpar.vmax,
           cmap = 'binary',
           aspect = 'auto')

  if fbflag == 1 :

    try :
      ax.plot( d.fbreak.time[ d.isrc, : ], rz, '.', markersize=3 )
    except :
      ax.plot( d.fbreak.time, rz, '.', markersize=3 )


  axpar.format_axes( ax )


def image_shot_gather_spectra_cw( ax, d, axpar, dflag='md') :



  if dflag == 'md' :
    rz = d.rcvs.id.astype( np.float ) 
  else :
    rz = d.rcvs.z

  ax.imshow( d.fdata / np.max(d.fdata) ,
           extent = ( d.f[0], d.f[-1], rz[-1], rz[0] ),
           cmap = 'inferno',
           aspect = 'auto')

  axpar.format_axes( ax )

def image_receiver_gather_cw( ax, d, axpar, dflag='md', fbflag=1 ) :
  if axpar.vmin is None :
    axpar.vmin = -0.8 * np.max( np.abs( d.data ) )

  if axpar.vmax is None :
    axpar.vmax = 0.8 * np.max( np.abs( d.data ) )

  if dflag == 'md' :
    sz = d.srcs.id.astype( np.float ) 
  else :
    sz = d.srcs.z

  ax.imshow( d.data, extent = ( d.t[0], d.t[-1], sz[-1], sz[0] ),
           vmin = axpar.vmin, vmax = axpar.vmax,
           cmap = 'binary',
           aspect = 'auto')

  if fbflag == 1 :
    try :
      ax.plot( d.fbreak.time[ :, d.ircv ], sz, '.', markersize=3 )
    except :
      ax.plot( d.fbreak.time, sz, '.', markersize=3 )


  axpar.format_axes( ax )


def image_receiver_gather_spectra_cw( ax, d, axpar, dflag='md' ) :



  if dflag == 'md' :
    sz = d.srcs.id.astype( np.float ) 
  else :
    sz = d.srcs.z

  ax.imshow( d.fdata / np.max( d.fdata ), 
           extent = ( d.f[0], d.f[-1], sz[-1], sz[0] ),
           cmap = 'inferno',
           aspect = 'auto')

  axpar.format_axes( ax )


def image_cmp_gather_cw( ax, d, axpar, fbflag=1 ) :
  if axpar.vmin is None :
    axpar.vmin = -0.8 * np.max( np.abs( d.data ) )

  if axpar.vmax is None :
    axpar.vmax = 0.8 * np.max( np.abs( d.data ) )

  sz = d.zoffset

  ax.imshow( d.data, extent = ( d.t[0], d.t[-1], sz[-1], sz[0] ),
           vmin = axpar.vmin, vmax = axpar.vmax,
           cmap = 'binary',
           aspect = 'auto')

  if fbflag == 1 :
    ax.plot( d.fbreak.time, sz, '.', markersize=3 )


  axpar.format_axes( ax )


def image_offset_gather_cw( ax, d, axpar, fbflag=1 ) :
  if axpar.vmin is None :
    axpar.vmin = -0.8 * np.max( np.abs( d.data ) )

  if axpar.vmax is None :
    axpar.vmax = 0.8 * np.max( np.abs( d.data ) )

  sz = d.zcmp

  ax.imshow( d.data, extent = ( d.t[0], d.t[-1], sz[-1], sz[0] ),
           vmin = axpar.vmin, vmax = axpar.vmax,
           cmap = 'binary',
           aspect = 'auto')

  if fbflag == 1 :
    ax.plot( d.fbreak.time, sz, '.', markersize=3 )


  axpar.format_axes( ax )


