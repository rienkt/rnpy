#!/usr/bin/env python
from __future__ import print_function
"""


  TIME.py

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
import datetime
import pytz
#import scipy as sp
#import matplotlib.pyplot as plt



#======================================================================
# classes / functions
#======================================================================
def julianday(t):

    tt = t.timetuple()
    return tt.tm_yday

def sec_from_epoc(t) :
    tepoc = datetime.datetime( 1970, 1, 1, tzinfo=pytz.utc )
    
    return ( t - tepoc ).total_seconds()

def utc( t, tz = None) :
    if tz is not None :
      t = tz.localize( pytz.timezone(t) )

    return t.astimezone( pytz.utc )

def julianday2date( yday, year, tz = None, hh=0, mm=0, ss=0, micros=0 ) :
  if tz is None :
    tz = pytz.utc
  
  try :
    date = datetime.datetime.strptime( '%d%d'%(year, yday), '%Y%j' )
  except :
    date = datetime.datetime.strptime( '%d%d'%(year, yday), '%y%j' )

  date = date.timetuple()

  return tz.localize(datetime.datetime( date.tm_year, date.tm_mon, date.tm_mday, hh, mm, ss, micros ))

#####

def epocsec2jp( sec ) :
    tutc   = datetime.datetime.utcfromtimestamp( sec )
    tz_jp  = pytz.timezone('Asia/Tokyo')
    tz_utc = pytz.utc
    tjp    = tz_utc.localize(tutc).astimezone( tz_jp ) 
    return tjp



