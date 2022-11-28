#!/usr/bin/env python
from __future__ import print_function
"""

  geometry.py

  Handle geomtery related scripts


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

import numpy as np
import pandas as pd
#import scipy as sp
#import matplotlib.pyplot as plt
#import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process
import utm

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================


def dist( pt1, pt2 ):
  if type(pt1) is list :
    pt1 = np.asarray( pt1, dtype=np.float )
  if type(pt2) is list :
    pt2 = np.asarray( pt2, dtype=np.float )
  return np.sqrt( np.sum( ( pt1-pt2 ) **2 ) )

def dist_xyz( x, y, z=None ):
  if z is None :
    return np.sqrt( (x[0]-x[1])**2 + (y[0]-y[1])**2 )
  else :
    return np.sqrt( (x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2 )

def rotate_xy( xin, yin, rot ) :
  crot = np.exp( 1j * rot )
  xout = np.real( ( xin + 1j*yin ) * crot ) 
  yout = np.imag( ( xin + 1j*yin ) * crot ) 
  return xout, yout

def to_latlon( utmx, utmy, izone, czone ) :
  print( izone, czone )
  if type( utmx ) is np.ndarray  :
    n = utmx.size
    lat = np.zeros( n, dtype=np.float )
    lon = np.zeros( n, dtype=np.float )
    print(n)
    for i in range(n) :
      tmp = utm.to_latlon( utmx[i], utmy[i], izone, czone )
      lat[i] = tmp[0]
      lon[i] = tmp[1]
  else :
    tmp = utm.to_latlon( utmx, utmy, izone, czone )
    lat = tmp[0]
    lon = tmp[1]
  return lat, lon


def from_latlon( lat, lon, izone=None, czone=None ) :
  if ( type( lat ) is np.ndarray  ) or ( type(lat) is pd.core.series.Series) : 
    n = lat.size
    utmx = np.zeros( n, dtype=np.float )
    utmy = np.zeros( n, dtype=np.float )
    print(n)
    for i in range(n) :
      tmp = utm.from_latlon( lat[i], lon[i], izone, czone )
      utmx[i] = tmp[0]
      utmy[i] = tmp[1]
  else :
    tmp = utm.from_latlon( lat, lon, izone, czone )
    utmx = tmp[0]
    utmy = tmp[1]
  return utmx, utmy


def utm2latlon( utmx, utmy, izone, czone ) :
  return to_latlon( utmx, utmy, izone, czone )

def latlon2utm( lat, lon, izone=None, czone=None ) :
  return from_latlon( lat, lon, izone, czone )
