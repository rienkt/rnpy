#!/usr/bin/env python
from __future__ import print_function
"""

  ELASTIC.py

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt



#======================================================================
# classes / functions
#======================================================================


def velocity_dispersion( vo, f, fo, q ) :
  qinv = 1./q
  return vo * ( 1. + qinv /np.pi * np.log( f/fo )   )


def vp2den_gardner( vp, vp_unit='m/s' ) :
  if vp_unit == 'km/s' :
    den = 0.31 * ( 1000.0 * vp ) **0.25
  else :
    den = 0.31 * ( vp ) **0.25
  return den

