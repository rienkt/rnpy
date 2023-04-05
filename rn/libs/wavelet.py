#!/usr/bin/env python
from __future__ import print_function
"""


  WAVELET.py

  filters related to wavelet or waveforms

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

import rk.libs.fourier as rk_fourier

#======================================================================
# classes / functions
#======================================================================

def phase_shift( d, t, deg, rad=None ) : 

  if not rad : 
    rad = np.deg2rad( deg )

  fd, f = rk_fourier.rk_rfft( d, t=t ) 

  crot = np.exp( -1j*rad*np.ones_like( fd ) )

  fd *= crot

  dout = rk_fourier.rk_irfft( fd )
  return dout



