#!/usr/bin/env python
"""

  OFFSET 

  Calculate 2D and 3D offset


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np


#======================================================================
# classes / functions
#======================================================================

def offset2d( sx, sy, rx, ry ) :
  rrx, ssx = np.meshgrid( rx, sx )
  rry, ssy = np.meshgrid( ry, sy )
  return np.sqrt( ( rrx - ssx ) ** 2 + ( rry - ssy ) ** 2 )


def offset3d( sx, sy, sz,  rx, ry, rz ) :
  rrx, ssx = np.meshgrid( rx, sx )
  rry, ssy = np.meshgrid( ry, sy )
  rrz, ssz = np.meshgrid( rz, sz )
  return np.sqrt( ( rrx - ssx ) ** 2 + ( rry - ssy ) ** 2 
                + ( rrz - ssz ) ** 2 )
