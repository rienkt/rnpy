#!/usr/bin/env python
from __future__ import print_function
"""

  BEACHBALL.py

  draw beach ball

  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import matplotlib.pyplot as plt


#======================================================================
# classes / functions
#======================================================================



def dc( ax, theta, r=1., ox=0., oz=0., color='k' ) :

  rads = np.arange( 0., np.pi/2., 0.01, dtype=np.float ) -np.pi/4.+ theta



  dcx1 = np.concatenate( ( [ ox ], r * np.cos( rads ) + ox, [ ox ] ) )
  dcz1 = np.concatenate( ( [ oz ], r * np.sin( rads ) + oz, [ oz ] ) )

  rads = np.arange( 0., np.pi/2., 0.01, dtype=np.float ) + np.pi*3./4.+ theta

  dcx2 = np.concatenate( ([ ox ], r * np.cos( rads ) + ox, [ ox ] ) )
  dcz2 = np.concatenate( ([ oz ], r * np.sin( rads ) + oz, [ oz ] ) )

  ax.fill( dcx1, dcz1, color, dcx2, dcz2, color )

  rads = np.arange( -np.pi, np.pi, 0.01, dtype=np.float )
  ax.plot( np.cos( rads ) * r + ox, np.sin( rads ) * r + oz, color=color )

 
 
def iso( ax,  r=1., ox=0., oz=0., color='k' ) :

  rads = np.arange( -np.pi, np.pi, 0.01, dtype=np.float )
  ax.plot( np.cos( rads ) * r + ox, np.sin( rads ) * r + oz , color=color)

 
 




