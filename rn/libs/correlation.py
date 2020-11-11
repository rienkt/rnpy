#!/usr/bin/env python
"""

  Correlation routines

  <ADD_DESCRIPTION_HERE>


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

def xcorr_fd( d1, d2, dt) :
     
  nt = d1.shape[-1]

  f = np.arange( 0, nt/2+1, dtype=np.float )  / dt / float( nt )

  fd1 = np.fft.rfft( d1, axis = -1 ) 
  fd2 = np.fft.rfft( d2, axis = -1 ) 


  return fd1*np.conj(fd2), f

