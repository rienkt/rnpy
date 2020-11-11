#!/usr/bin/env python
from __future__ import print_function
"""
  pad2d.py npad=int zsea=float
    extending npad grids at the top and bottom  of the model
    also set the same velocity from the top to zsea 
"""
#==============================================================================
#                                                                IMPORT MODULES 
#==============================================================================
import numpy as np

# RK module
from rk.rsf.model import Model

#===============================================================================
#                                                             FUNCTIONS/CLASSES
#==============================================================================

def pad2d( m, npad, val=None)  :



  mpad = Model( oz = m.oz - npad * m.dz, dz = m.dz, nz = m.nz + 2 * npad, 
                ox = m.ox - npad * m.dx, dx = m.dx, nx = m.nx + 2 * npad)
  if val :
    mpad.initialize( val )


  mpad.data[ npad:-npad,        npad:-npad ] = m.data
  if val is None :
    mpad.data[ :npad,             npad:-npad ] = m.data[ 0,  : ]
    mpad.data[ mpad.nx - npad : , npad:-npad ] = m.data[ -1, : ]
    mpad.data[ :,                     :npad  ] = np.repeat(
                                                mpad.data[ :,  npad:npad+1 ], 
                                                npad, axis=1 )
    mpad.data[ :, mpad.nz - npad :  ]          = np.repeat(
                                 mpad.data[ :,  mpad.nz-npad-1:mpad.nz-npad], 
                                 npad, axis=1 )


  return mpad

