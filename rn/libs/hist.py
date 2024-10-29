#!/usr/bin/env python
from __future__ import print_function
"""


  <MODULE NAME>

  <ADD_DESCRIPTION_HERE>


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

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================

class binned_data () :
  def __init__ ( self, dbin=1., obin=0., nbin=1, bin0=None, bin1=None ) :
    self.dbin = dbin
    self.obin = obin
    self.nbin = nbin
    #print( bin0 )
    if bin0 is not None :
      #print( 'here' )
      self.obin = bin0
      self.bins = np.arange( bin0, bin1, dbin )
      #print( self.bins )
      self.nbin = self.bins.shape[0]
    else :
      self.bins = np.arange( self.nbin ) * self.dbin + self.obin
    self.initialize()
  def initialize( self, val=0 )  :
    #self.d = np.ones( self.nbin, dtype=float ) * val
    self.mean = np.ones( self.nbin, dtype=float ) * val
    self.logmean = np.ones( self.nbin, dtype=float ) * val
    self.logstd = np.ones( self.nbin, dtype=float ) * val
    self.std = np.ones( self.nbin, dtype=float ) * val
    self.median = np.ones( self.nbin, dtype=float ) * val
    self.nbins = np.ones( self.nbin, dtype=int )
  def calc_stats( self, data, x ) :
    ibins = ( ( x- self.obin ) / self.dbin ).astype( 'int' )
    #print( ibins[100:105] )
    for ibin in range( self.nbin ) :
      print( ibin )
      idxs = np.where( ibins ==ibin ) 
      print( idxs )
      mydata =  data[idxs] 
      mydata = mydata.compressed()
      #print( mydata )
      mydata = mydata[~np.isnan(mydata)]
      mydata = mydata[np.abs(mydata)>1e-20] 
      self.nbins[ibin] = mydata.size
      if mydata.size > 0 :
        self.mean[ibin] = np.ma.mean(mydata ) 
        self.std[ibin] = np.ma.std(mydata ) 
        self.median[ibin] = np.ma.median(mydata ) 
        if min(mydata) > 0 :
        # the next line will cuase a problem if mydata contains negative valuye
          self.logmean[ibin] = np.ma.exp( np.ma.mean(np.ma.log(mydata ) ) )
          self.logstd[ibin] = np.ma.std( np.ma.log( mydata )  )
        else :
          print( 'we do not compute logmean and logstd, as min(mydata) < 0')
    self.mean = np.ma.masked_equal( self.mean, 0 )
    self.median = np.ma.masked_equal( self.median, 0 )
    self.logmean = np.ma.masked_equal( self.logmean, 0 )
    self.logstd = np.ma.masked_equal( self.logstd, 0 )
    self.std = np.ma.masked_equal( self.std, 0 )
    #print( 'standard', self.std )
 
  def write( self, f ) :
    df = pd.DataFrame({ 'bin': self.bins, 'mean': self.mean,
                  'median': self.median, 'std': self.std,
                  'logmean': self.logmean, 'logstd':self.logstd } )
    df.to_csv( f )
  def read( self, f ) :
    print( f )
    df = pd.read_csv( f )
    self.bins = df.bin.to_numpy()
    self.mean = df['mean'].to_numpy()
    self.median = df['median'].to_numpy()
    self.logmean = df['logmean'].to_numpy()
    self.logstd = df['logstd'].to_numpy()
    try :
      self.std = df['std'].to_numpy()
    except :
      self.std = 0.



class binned_data_2d () :
  def __init__ ( self, 
          dbin0=1., obin0=0., nbin0=1, bin0_0=None, bin0_1=None,
          dbin1=1., obin1=0., nbin1=1, bin1_0=None, bin1_1=None,
          ) :
    self.dbin0 = dbin
    self.obin0 = obin
    self.nbin0 = nbin
    #print( bin0 )
    if bin0 is not None :
      #print( 'here' )
      self.obin = bin0
      self.bins = np.arange( bin0, bin1, dbin )
      #print( self.bins )
      self.nbin = self.bins.shape[0]
    else :
      self.bins = np.arange( self.nbin ) * self.dbin + self.obin
    self.initialize()
  def initialize( self, val=0 )  :
    #self.d = np.ones( self.nbin, dtype=float ) * val
    self.mean = np.ones( self.nbin, dtype=float ) * val
    self.logmean = np.ones( self.nbin, dtype=float ) * val
    self.logstd = np.ones( self.nbin, dtype=float ) * val
    self.median = np.ones( self.nbin, dtype=float ) * val
  def calc_stats( self, data, x ) :
    ibins = ( ( x- self.obin ) / self.dbin ).astype( 'int' )
    print( ibins[0:10] )
    for ibin in range( self.nbin ) :
      idxs = np.where( ibins ==ibin ) 
      mydata = data[idxs]
      mydata = mydata[~np.isnan(mydata)]
      mydata = mydata[mydata>0] 
      self.mean[ibin] = np.mean(mydata ) 
      self.median[ibin] = np.median(mydata ) 
      self.logmean[ibin] = np.exp( np.mean(np.log(mydata ) ) )
      self.logstd[ibin] = np.std( np.log( mydata )  )


class binned_data_sliding( binned_data ) :
  def __init__ ( self, dbin=1., obin=0., nbin=1, binsize=None,
                 bin0=None, bin1=None ) :
    self.dbin = dbin
    self.obin = obin
    self.nbin = nbin
    if binsize is None :
      binsize = self.dbin
    self.binsize = binsize

    if bin0 is not None :
      self.obin = bin0
      self.bins = np.arange( bin0, bin1, dbin )
      self.nbin = self.bins.shape[0]
    else :
      self.bins = np.arange( self.nbin ) * self.dbin + self.obin
    self.bins0 = self.bins - self.binsize/2.
    self.bins1 = self.bins + self.binsize/2.
    self.initialize()
  def calc_stats( self, data, x ) :
    #ibins = ( ( x- self.obin ) / self.dbin ).astype( 'int' )
    for ibin in range( self.nbin ) :
      #print( self.bins0[ibin]  )
      #print( x )
      #print( np.where( x > self.bins0[ibin] ) )
      idxs = np.where(  (x > self.bins0[ibin] )  
                          & ( x <= self.bins1[ibin] )  )
      #print( idxs, data.shape )
      mydata =  data[idxs] 
      mydata = mydata.compressed()
      #print( mydata )
      mydata = mydata[~np.isnan(mydata)]
      mydata = mydata[np.abs(mydata)>1e-20] 
      self.nbins[ibin] = mydata.size
      if mydata.size > 0 :
        self.mean[ibin] = np.ma.mean(mydata ) 
        self.std[ibin] = np.ma.std(mydata ) 
        self.median[ibin] = np.ma.median(mydata ) 
        if min(mydata) > 0 :
        # the next line will cuase a problem if mydata contains negative valuye
          self.logmean[ibin] = np.ma.exp( np.ma.mean(np.ma.log(mydata ) ) )
          self.logstd[ibin] = np.ma.std( np.ma.log( mydata )  )
        else :
          print( 'we do not compute logmean and logstd, as min(mydata) < 0')
    self.mean = np.ma.masked_equal( self.mean, 0 )
    self.median = np.ma.masked_equal( self.median, 0 )
    self.logmean = np.ma.masked_equal( self.logmean, 0 )
    self.logstd = np.ma.masked_equal( self.logstd, 0 )
    self.std = np.ma.masked_equal( self.std, 0 )
    #print( 'standard', self.std )
 
  def calc_stats_n( self, data, x ) : # data.ndim = x.ndim + 1
    #ibins = ( ( x- self.obin ) / self.dbin ).astype( 'int' )
    for ibin in range( self.nbin ) :
      #print( self.bins0[ibin]  )
      #print( x )
      #print( np.where( x > self.bins0[ibin] ) )
      ixs, iys = np.where(  (x > self.bins0[ibin] )  
                          & ( x <= self.bins1[ibin] )  )
      #print( x.shape )
      #print( data.shape )
      #print( ixs, iys )
      mydata =  data[:,ixs, iys] 
      mydata = mydata.compressed()
      #print( mydata )
      mydata = mydata[~np.isnan(mydata)]
      mydata = mydata[np.abs(mydata)>1e-20] 
      self.nbins[ibin] = mydata.size
      if mydata.size > 0 :
        self.mean[ibin] = np.ma.mean(mydata ) 
        self.std[ibin] = np.ma.std(mydata ) 
        self.median[ibin] = np.ma.median(mydata ) 
        if min(mydata) > 0 :
        # the next line will cuase a problem if mydata contains negative valuye
          self.logmean[ibin] = np.ma.exp( np.ma.mean(np.ma.log(mydata ) ) )
          self.logstd[ibin] = np.ma.std( np.ma.log( mydata )  )
        else :
          print( 'we do not compute logmean and logstd, as min(mydata) < 0')
    self.mean = np.ma.masked_equal( self.mean, 0 )
    self.median = np.ma.masked_equal( self.median, 0 )
    self.logmean = np.ma.masked_equal( self.logmean, 0 )
    self.logstd = np.ma.masked_equal( self.logstd, 0 )
    self.std = np.ma.masked_equal( self.std, 0 )
    #print( 'standard', self.std )
