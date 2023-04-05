#!/usr/bin/env python
"""
  module wirk.rsf.model :
    Define Model Class

"""
# Import RSF structure (like SEP.top)
import rsf.api as rsf
import os 
import sys
import subprocess
import copy

# Import math packages
import numpy as np
import scipy as sp
from math import sqrt

from .fd import rn_freq
from rn.libs.misc import eprint
#================================================================
# define class
#================================================================

class Model:
  def __init__(self, ref=None, 
                     oz=0., dz=1., nz=1,
                     ox=0., dx=1., nx=1, init_value=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
    else :
      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
    self.initialize( init_value )
    self.set_axis()

  def initialize( self, init_value=0.0 ):
    self.d = np.ones( ( self.nx, self.nz ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initialize( init_value  )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.nz = input.int(   'n1' )
    self.nx = input.int(   'n2' )
    self.dz = input.float( 'd1' )
    self.dx = input.float( 'd2' )
    self.oz = input.float( 'o1' )
    self.ox = input.float( 'o2' )
    self.set_axis()
 

  def read( self, input=None, f=None ):
    flag = 0
# def read( self, input=None, f=None, fbin=None ):
    if input is None :
      try :
        input = rsf.Input( f )
      except :
        print( 'file %f cannot be found'%f )
      #flag = 1


    try :
      self.read_header( input=input ) 
      #print( self.dx )

      self.initialize( 0.0 )
      input.read(self.d)
      self.set_axis()
      #if flag == 1 :
      #  input.close()
    except :
      print( 'usage: self.read( input=hoge, f=hoge.rsf )' )
      print( '       we need either of input or file name' )

  def write( self, output=None, f=None, fsrc=None ) :
    #print( 'here model 1', output, f, 'fsrc',  fsrc )
    if output is None :
      if fsrc :
        output = rsf.Output( f, src=rsf.Input( fsrc ) )
      else :
        #output = rsf.Output( tag='test.rsf' )
        output = rsf.Output( f ) #, src=rsf.Input('vp-00.rsf') )
    #print( 'here model 2' )

    output.put( 'n1', self.nz )
    output.put( 'n2', self.nx )
    output.put( 'd1', self.dz )
    output.put( 'd2', self.dx )
    output.put( 'o1', self.oz )
    output.put( 'o2', self.ox )
    #eprint( self.d.max() )
    #eprint( self.d.shape )
    output.write( self.d.astype( np.float32 ) )
    output.close()
    #output.close()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz

  def norm( self ):
    norm = np.dot( self.d.reshape( ( self.nx * self.nz, 1 ) )
                       [ 0:self.nx * self.nz, 0 ], 
                   self.d.reshape( ( self.nx * self.nz, 1 ) ).T
                       [ 0, 0 : self.nx * self.nz ] )
    norm = sqrt( norm )
    return norm
     
  def normalize( self ):
    self.d /= np.abs( self.d ).max()

  def m2km( self ) :
    self.d *= 1e-3
    self.x *= 1e-3
    self.z *= 1e-3


class CModel( Model ) :
  def initialize( self, init_value = 0.0 ) :
    self.d = np.ones( ( self.nx, self.nz ), dtype=np.complex64 )  * init_value
  def initialise( self, init_value = 0.0 ) :
    self.initialize( init_value )

import copy
class FreqDomainModelSnap :
  def __init__(self, ref=None, 
                oz=0., dz=1., nz=1,
                ox=0., dx=1., nx=1, 
                nfreq=1, init_value=0.):
    self.freqs = rn_freq()
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
      self.freqs = ref.freqs
    else :
      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
      self.freqs.n = nfreq

    self.initialize( init_value )
  def initialize( self, init_value=0.0 ):
    self.d = np.ones( ( self.freqs.n, self.nx, self.nz ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initialize( init_value  )


  def read_header( self, input_re=None, input_im=None,
                  fre=None, fim=None,  ffreq='freq.txt' ) :
    if input_re is None :
      input_re = rsf.Input( fre )
    self.nfreq = input_re.int(   'n3' )
    self.nz = input_re.int(   'n1' )
    self.nx = input_re.int(   'n2' )
    self.dz = input_re.float( 'd1' )
    self.dx = input_re.float( 'd2' )
    self.oz = input_re.float( 'o1' )
    self.ox = input_re.float( 'o2' )

    if os.path.exists( ffreq ) :
      self.freqs.read( ffreq )
      if self.freqs.n != self.nfreq :
        print( '# of frequencies differs between freq file and rsf file')

  def read( self, input_re=None, input_im=None, fre=None, fim=None,
            ffreq='freq.txt'):

    flag_re = 0
    if input_re is None :
      input_re=rsf.Input(fre)
      flag_re = 1
    flag_im = 0
    if input_im is None :
      input_im=rsf.Input(fim)
    flag_im = 1
    self.read_header( input_re=input_re, input_im=input_im,
                      ffreq=ffreq )

    tmp_re=np.zeros( ( self.freqs.n, self.nx, self.nz ),'f')
    tmp_im=np.zeros( ( self.freqs.n, self.nx, self.nz ),'f')
    input_re.read( tmp_re )
    input_im.read( tmp_im )
    self.d = tmp_re + 1j * tmp_im
    self.set_axis()

    if flag_re == 1 :
      input_re.close()
    if flag_im == 1 :
      input_im.close()

  def write( self, output=None, f=None ) :
    if output is None :
      output = rsf.Output( f )

    output.put( 'n1', self.nz )
    output.put( 'n2', self.nx )
    output.put( 'n3', self.freqs.n )
    output.put( 'd1', self.dz )
    output.put( 'd2', self.dx )
    output.put( 'd3', 1. )
    output.put( 'o1', self.oz )
    output.put( 'o2', self.ox )
    output.put( 'o2', self.ox )
    output.write( self.d.astype( np.float32 ) )
    output.close()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz


  def m2km( self ) :
    self.x *= 1e-3
    self.z *= 1e-3

class TimeDomainModelSnap :
  def __init__fd(self, ref=None, 
                oz=0., dz=1., nz=1,
                ox=0., dx=1., nx=1, 
                ot=0., dt=1., nt=1, init_value=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
      self.nt = ref.nt
      self.dt = ref.dt
      self.ot = ref.ot
    else :
      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
      self.nt = nt
      self.dt = dt
      self.ot = ot
    self.initialize( init_value )

  def initialize( self, init_value=0.0 ):
    self.d = np.ones( ( self.nx, self.nz, self.nt ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initialize( init_value  )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.nt = input.int(   'n1' )
    self.nz = input.int(   'n2' )
    self.nx = input.int(   'n3' )
    self.dt = input.float( 'd1' )
    self.dz = input.float( 'd2' )
    self.dx = input.float( 'd3' )
    self.ot = input.float( 'o1' )
    self.oz = input.float( 'o2' )
    self.ox = input.float( 'o3' )
 

  def read( self, input=None, f=None ):
    flag = 0
# def read( self, input=None, f=None, fbin=None ):
    if input is None :
      try :
        input = rsf.Input( f )
      except :
        print( 'file %f cannot be found'%f )
      flag = 1

    self.read_header( input=input ) 
    print( self.dx )

    self.initialize( 0.0 )
    input.read(self.d)
    self.set_axis()
    if flag == 1 :
      input.close()

  def write( self, output=None, f=None ) :
    if output is None :
      output = rsf.Output( f )

    output.put( 'n1', self.nz )
    output.put( 'n2', self.nx )
    output.put( 'd1', self.dz )
    output.put( 'd2', self.dx )
    output.put( 'o1', self.oz )
    output.put( 'o2', self.ox )
    output.write( self.d.astype( np.float32 ) )
    output.close()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz
    self.t = np.arange( 0, self.nt, dtype=float ) * self.dt + self.ot

     

  def m2km( self ) :
    self.d *= 1e-3
    self.x *= 1e-3
    self.z *= 1e-3

class MCTimeDomainModelSnap :
  def __init__fd(self, ref=None, 
                nc=1,
                oz=0., dz=1., nz=1,
                ox=0., dx=1., nx=1, 
                ot=0., dt=1., nt=1, init_value=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
      self.nt = ref.nt
      self.dt = ref.dt
      self.ot = ref.ot
      self.nc = ref.nc

    else :
      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
      self.nt = nt
      self.dt = dt
      self.ot = ot
      self.nc = nc
    self.initialize( init_value )

  def initialize( self, init_value=0.0 ):
    self.d = np.ones( ( self.nt, self.nc, self.nx, self.nz ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initialize( init_value  )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.nt = input.int(   'n4' )
    self.nz = input.int(   'n1' )
    self.nx = input.int(   'n2' )
    self.nc = input.int( 'n3' )
    self.dt = input.float( 'd4' )
    self.dz = input.float( 'd1' )
    self.dx = input.float( 'd2' )
    self.ot = input.float( 'o4' )
    self.oz = input.float( 'o1' )
    self.ox = input.float( 'o2' )
 

  def read( self, input=None, f=None ):
    flag = 0
# def read( self, input=None, f=None, fbin=None ):
    if input is None :
      try :
        input = rsf.Input( f )
      except :
        print( 'file %f cannot be found'%f )
      flag = 1

    self.read_header( input=input ) 
    #print( self.dx )

    self.initialize( 0.0 )
    input.read(self.d)
    self.set_axis()
    if flag == 1 :
      input.close()

  def write( self, output=None, f=None ) :
    if output is None :
      output = rsf.Output( f )
    print( 'this option is not implemented yet')
    #output.put( 'n1', self.nz )
    #output.put( 'n2', self.nx )
    #output.put( 'd1', self.dz )
    #output.put( 'd2', self.dx )
    #output.put( 'o1', self.oz )
    #output.put( 'o2', self.ox )
    #output.write( self.d.astype( np.float32 ) )
    #output.close()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz
    self.t = np.arange( 0, self.nt, dtype=float ) * self.dt + self.ot

     

  def m2km( self ) :
    self.d *= 1e-3
    self.x *= 1e-3
    self.z *= 1e-3

