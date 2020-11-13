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
    self.initialize( 0.0 )

  def initialize( self, init_value=0.0 ):
    self.d = np.ones( ( self.nx, self.nz ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initialize( 0.0 )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.nz = input.int(   'n1' )
    self.nx = input.int(   'n2' )
    self.dz = input.float( 'd1' )
    self.dx = input.float( 'd2' )
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
    self.x = np.arange( 0, self.nx, dtype=np.float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=np.float ) * self.dz + self.oz

  def norm( self ):
    norm = np.dot( self.d.reshape( ( self.nx * self.nz, 1 ) )
                       [ 0:self.nx * self.nz, 0 ], 
                   self.d.reshape( ( self.nx * self.nz, 1 ) ).T
                       [ 0, 0 : self.nx * self.nz ] )
    norm = sqrt( norm )
    return norm
     

  def m2km( self ) :
    self.d *= 1e-3
    self.x *= 1e-3
    self.z *= 1e-3
