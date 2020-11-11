#!/usr/bin/env python

# Import RSF structure (like SEP.top)
import rsf.api as rsf
#import m8r as sf
import sys
import subprocess
import os.path
# Import math packages
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg') # this will disable X connection
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from math import sqrt

     
# define function     
   
def get_dim(input):
    dim=0
    for i in range(1,10):
        if input.int('n'+str(i))==None:
            break
        else:
            dim=i
    return dim


class RealData:
  def __init__(self,ref=None,orcv=0., drcv=1., nrcv=1, osrc=0., dsrc=1., nsrc=1, ofreq=0., dfreq=1., nfreq=1, oiter=0., diter=1., niter=1):
    if ref==None:
      self.orcv=orcv
      self.drcv=drcv
      self.nrcv=nrcv
      self.osrc=osrc
      self.dsrc=dsrc
      self.nsrc=nsrc
      self.ofreq=ofreq
      self.dfreq=dfreq
      self.nfreq=nfreq
      self.oiter=oiter
      self.diter=diter
      self.niter=niter
    else:
      self.orcv=ref.orcv
      self.drcv=ref.drcv
      self.nrcv=ref.nrcv
      self.osrc=ref.osrc
      self.dsrc=ref.dsrc
      self.nsrc=ref.nsrc
      self.ofreq=ref.ofreq
      self.dfreq=ref.dfreq
      self.nfreq=ref.nfreq
      self.oiter=ref.oiter
      self.diter=ref.diter
      self.niter=ref.niter
  def initialize(self,gather='all',init_value=0.):
    if gather=='s':
      self.data=np.ones((self.nfreq,self.nrcv),'f')*init_value
    elif gather=='r':
      self.data=np.ones((self.nfreq,self.nsrc),'f')*init_value
    elif gather=='f':
      self.data=np.ones((self.nsrc,self.nrcv),'f')*init_value
    else:
      self.data=np.ones((self.nfreq,self.nsrc,self.nrcv),'f')*init_value
  def initialise(self,gather='all',init_value=0.):
    self.initialize(gather, init_value)

  def read_header(self, fre, 
                  sfile = 'src.dat', rfile = 'rcv.dat', ffile = 'freq.dat'):
    self.input = rsf.Input(fre)

    self.nrcv = self.input.int('n1')
    self.nsrc = self.input.int('n2')
    self.drcv = self.input.float('d1')
    self.dsrc = self.input.float('d2')
    self.orcv = self.input.float('o1')
    self.osrc = self.input.float('o2')

    if (get_dim(self.input) >= 3):
      self.nfreq = self.input.int('n3')
      self.dfreq = self.input.float('d3')
      self.ofreq = self.input.float('o3')
    else:
      self.nfreq = 1
      self.dfreq = 1.
      self.ofreq = 1.

    if (get_dim(self.input) >= 4):
      self.niter = self.input.int('n4')
      self.diter = self.input.float('d4')
      self.oiter = self.input.float('o4')
    else:
      self.niter = 1
      self.diter = 1.
      self.oiter = 1.


    # define axis
    if os.path.isfile(sfile):
      self.srcx = np.fromfile( sfile, sep = ' ', count = self.nsrc * 2
                  ).reshape( [self.nsrc, 2] )[:, 0]
      self.srcz = np.fromfile( sfile, sep = ' ', count = self.nsrc * 2 
                  ).reshape( [self.nsrc, 2] )[:, 1]
    else:
      self.srcx = np.arange( 0, self.nsrc, dtype = 'f' )
      self.srcz = np.arange( 0, self.nsrc, dtype = 'f' )

    if os.path.isfile(rfile):
      self.rcvx = np.fromfile( rfile, sep = ' ', count = self.nrcv * 2
                ).reshape( [self.nrcv, 2] )[:, 0]
      self.rcvz = np.fromfile( rfile, sep = ' ', count = self.nrcv * 2
                ).reshape( [self.nrcv, 2] )[:, 1]
    else:
      self.rcvx=np.arange(0., self.nrcv, dtype = 'f')
      self.rcvz=np.arange(0., self.nrcv, dtype = 'f')

    if os.path.isfile(ffile):
      self.freq = np.fromfile( ffile, sep=' ', count = self.nfreq + 1
                )[ 1 : ( self.nfreq + 1)]
    else :
      self.freq = ( np.arange(0, self.nfreq, dtype = 'f' ) * self.dfreq 
                    + self.ofreq )

    self.iter = np.arange(0, self.niter, dtype = 'f')


    try :
      self.srcx = self.srcx * self.scale
      self.srcz = self.srcz * self.scale
      self.rcvx = self.rcvx * self.scale
      self.rcvz = self.rcvz * self.scale
    except:
      print 'no scaling for axis'


  def read_rsf(self, fre, gather = 'all', inum = 1, 
              sfile = 'src.dat', rfile = 'rcv.dat', ffile = 'freq.dat'):

    self.read_header(fre, sfile = sfile, rfile = rfile, ffile = ffile )

    tmp_re = np.zeros( ( self.nsrc, self.nrcv ), 'f' )

    if gather=='s' : # shot gather at src # and iter 0 (assume ndim=<3)
      isrc=inum
      tmpdata_real=np.zeros((self.nfreq,self.nrcv),'f')
      for ifreq in range(self.nfreq):
        self.input.read(tmp_re)
        tmpdata_real[ifreq,:]=tmp_re[isrc-1,:]
    elif gather=='r' : # receiver gather at receiver # and iter 0 (assume ndim<=3) 
      ircv=inum
      tmpdata_real=np.zeros((self.nfreq,self.nsrc),'f')
      for ifreq in range(self.nfreq):
        self.input.read(tmp_re)
        tmpdata_real[ifreq,:]=tmp_re[:,ircv-1]

    elif gather=='f': # src-receiver map @ frequency # and iter 0 (assume ndim<=3)
      ifreq=inum
      tmpdata_real=np.zeros((self.nrcv,self.nsrc),'f')
      for i in range(self.nfreq):
        self.input.read(tmp_re)
        if i==(ifreq-1):
          self.input.read(tmpdata_real)
          break

    elif gather=='b': # block gather at receiver 0 and freq 0 (assume ndim=4)
      tmpdata_real=np.zeros((self.niter,self.nsrc),'f')
      for iiter in range(self.niter):
        for ifreq in range(self.nfreq):
          self.input.read(tmp_re)
          if ifreq==0:
            tmpdata_real[iiter,:]=tmp_re[:,0]

    else: 
      tmpdata_real=np.zeros((self.nfreq,self.nsrc,self.nrcv),'f')
      self.input.read(tmpdata_real)

    self.data=tmpdata_real

    

  def write_header(self,fre):
    self.output=rsf.Output(fre)
    self.output.put('n1',self.nrcv)
    self.output.put('n2',self.nsrc)
    self.output.put('n3',self.nfreq)
    self.output.put('n4',self.niter)
    self.output.put('d1',self.drcv)
    self.output.put('d2',self.dsrc)
    self.output.put('d3',self.dfreq)
    self.output.put('d4',self.diter)
    self.output.put('o1',self.orcv)
    self.output.put('o2',self.osrc)
    self.output.put('o3',self.ofreq)
    self.output.put('o4',self.diter)

  def write_rsf(self,fre):
    self.write_header(fre)
    self.output.write(np.float32(self.data))
#    output.write(self.data.transpose())
    self.output.close()

  def write_rsf_freq(self):
    self.output.write(np.float32(self.data))


  def write_close(self):
    self.output.close()
    
