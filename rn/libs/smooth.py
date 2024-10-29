#!/usr/bin/env python
"""

  SMOOTH.PY 

  Smoothing models


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
from scipy import ndimage


#======================================================================
# classes / functions
#======================================================================


def rn_moving( data, nsize ) :
  return ndimage.filters.uniform_filter( np.ma.masked_invalid(data), size = nsize, mode = 'nearest' )

def rn_gaussian( data, fsigma1, fsigma2 ) :
  return ndimage.filters.gaussian_filter( np.ma.masked_invalid(data), ( fsigma1, fsigma2 ) )
