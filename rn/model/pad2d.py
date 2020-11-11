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
# Import RSF structure (like SEP.top)
import rsf.api as rsf

# Import Math libraries

# RK module
from rk.rsf.model import Model

#===============================================================================
#                                                             FUNCTIONS/CLASSES
#==============================================================================

def pad2d( m, npad )  :



  mpad = Model( oz = m.oz - npad * m.dz, dz = m.dz, nz = m.nz + 2 * npad, 
                ox = m.ox - npad * m.dx, dx = m.dx, nx = m.nx + 2 * npad)


  mpad.d[ npad:-npad, npad:-npad ] = m.d

  mpad.d[ :npad, : ]  = mpad.d[ npad:npad+1,  : ]#.repeat( npad, axis=0 )
  mpad.d[ mpad.nx - npad:, :] = mpad.d[ -npad-1, : ]
  mpad.d[ :, :npad]           = mpad.d[ :,  npad:npad+1 ]
  mpad.d[ :, mpad.nz - npad :  ] = mpad.d[ :, -npad-1:-npad ]

  return mpad
