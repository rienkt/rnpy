#!/usr/bin/env python
"""

  MODEL.py

  define model class


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np

import sys, os 
import subprocess
import copy

# Import math packages
import numpy as np
import scipy as sp
from math import sqrt


#======================================================================
# classes / functions
#======================================================================

def read_from_textfile( ftxt ) :
  with open( ftxt, 'r' ) as f :
    lines = f.read().splitlines()
  return lines

def write_to_textfile( ftxt, outlines ) :
  with open( ftxt, 'w' ) as f :
    f.write( '\n'.join( outlines ) )
 

class Model:
#{{{{{
  def __init__(self, ref=None, 
                     oz=0., dz=1., nz=1,
                     ox=0., dx=1., nx=1, val=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
      try :
        self.fdir = ref.fdir
      except :
        print( 'fdir is not defined' )
      try :
        self.fheader = ref.fheader
      except :
        print( 'fheader is not defined' )
    else :

      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
    self.initialize( val=val )

    self.set_axis()


    self.fbin    = 'test.bin'
    self.fheader = 'test.header'


  def initialize( self, val=0.0 ):
    self.d = np.ones( ( self.nx, self.nz ), np.float32 
                       ) * val
    self.data = self.d

  def initialise( self, val=0.0 ):
    self.initialize( 0.0 )

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fheader = os.path.basename( fname )

  def read_header( self, fheader=None ) :
    if fheader :
      self.fheadeer = fheader

    self.set_fname( fheader )
    
    lines = read_from_textfile( os.path.join( self.fdir, self.fheader ) ) 

    # read x-coordinate
    iline = 0 
    ox, dx, nx = lines[ iline ].split()
    self.ox = float(ox) ; self.dx = float(dx); self.nx = int(nx)
    iline += 1
    oz, dz, nz = lines[ iline ].split()
    self.oz = float(oz) ; self.dz = float(dz); self.nz = int(nz)
    iline += 1
    self.fbin = lines[ iline ]

    self.set_axis()
  

  def read_bin( self ) :
    self.read_data()

  def read_data( self , vmin=-9999.) :
    print( vmin )
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.float32 
                  ).reshape( self.nx, self.nz ), vmin )
    self.data =self.d

  def read_data_fast( self, f=None ) :
    fbin = os.path.join( self.fdir, self.fbin )
    if f :
      fbin = f
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.int32 
                  ).reshape( self.nz, self.nx ), -9999. ).T.astype( np.float )
    self.data = self.d

  def read( self, fheader=None, vmin=-9999. ):
    self.read_header( fheader )
    self.read_data(vmin=vmin)

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fbin = fh + '.bin' 


  def write_header( self, fheader=None, fbin=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fbin :
      self.fbin = fbin



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oz, self.dz, self.nz ) )
    outlines.append( '%s'%self.fbin )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )

    try :
      self.d.filled( -9999. ).astype( np.float32 ).tofile( fbin )
    except :
      self.d.astype( np.float32 ).tofile( fbin )

  def write_data_int( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fbin )
    except :
      self.d.tofile( fbin )

  def write_data_fast( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.data.astype( np.int32 )

    try :
      self.d.T.filled( -9999 ).tofile( fbin )
    except :
      self.d.T.tofile( fbin )


  def write( self, fheader=None, fbin=None ) :
    self.write_header( fheader, fbin )
    self.write_data()

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
     
#}}}}}     
 

class Modelxy:
#{{{{{
  def __init__(self, ref=None, 
                     oy=0., dy=1., ny=1,
                     ox=0., dx=1., nx=1, val=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.ny = ref.ny
      self.oy = ref.oy
      self.dy = ref.dy
      self.fdir = ref.fdir
      self.fheader = ref.fheader
    else :

      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.ny = ny
      self.oy = oy
      self.dy = dy
    self.initialize( val=val )

    self.set_axis()


    self.fbin    = 'test.bin'
    self.fheader = 'test.header'


  def initialize( self, val=0.0 ):
    self.d = np.ones( ( self.nx, self.ny ), np.float32 
                       ) * val
    self.data = self.d

  def initialise( self, val=0.0 ):
    self.initialize( 0.0 )

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fheader = os.path.basename( fname )

  def read_header( self, fheader=None ) :
    if fheader :
      self.fheadeer = fheader

    self.set_fname( fheader )
    
    lines = read_from_textfile( os.path.join( self.fdir, self.fheader ) ) 

    # read x-coordinate
    iline = 0 
    ox, dx, nx = lines[ iline ].split()
    self.ox = float(ox) ; self.dx = float(dx); self.nx = int(nx)
    iline += 1
    oy, dy, ny = lines[ iline ].split()
    self.oy = float(oy) ; self.dy = float(dy); self.ny = int(ny)
    iline += 1
    self.fbin = lines[ iline ]

    self.set_axis()
  

  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.float32 
                 ).reshape( self.nx, self.ny ), -9999. )
    self.data =self.d


  def read( self, fheader=None ):
    self.read_header( fheader )
    self.read_data()

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fbin = fh + '.bin' 
    print( self.fbin )


  def write_header( self, fheader=None, fbin=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fbin :
      self.fbin = fbin



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oy, self.dy, self.ny ) )
    outlines.append( '%s'%self.fbin )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )

    try :
      self.d.filled( -9999. ).astype( np.float32 ).tofile( fbin )
    except :
      self.d.astype( np.float32 ).tofile( fbin )

  def write_data_int( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fbin )
    except :
      self.d.tofile( fbin )

  def write_data_fast( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.data.astype( np.int32 )

    try :
      self.d.T.filled( -9999 ).tofile( fbin )
    except :
      self.d.T.tofile( fbin )


  def write( self, fheader=None, fbin=None ) :
    self.write_header( fheader, fbin )
    self.write_data()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=np.float ) * self.dx + self.ox
    self.y = np.arange( 0, self.ny, dtype=np.float ) * self.dy + self.oy


  def extract( self, ix0, ix1, iy0, iy1 ) :
    m = Modelxy( dx=self.dx, dy=self.dy, nx=ix1-ix0, ny=iy1-iy0,
                ox=self.x[ix0], oy=self.y[iy0] )
    m.d = self.d[ ix0:ix1, iy0:iy1 ]
    return m
     
#}}}}}     


class Model3d:
  def __init__(self, ref=None, 
                     oz=0., dz=1., nz=1,
                     oy=0., dy=1., ny=1,
                     ox=0., dx=1., nx=1, val=0.):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nz = ref.nz
      self.oz = ref.oz
      self.dz = ref.dz
      self.ny = ref.ny
      self.oy = ref.oy
      self.dy = ref.dy
      self.fdir = ref.fdir
      self.fheader = ref.fheader
    else :

      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nz = nz
      self.oz = oz
      self.dz = dz
      self.ny = ny
      self.oy = oy
      self.dy = dy
    self.initialize( val=val )

    self.set_axis()


    self.fbin    = 'test.bin'
    self.fheader = 'test.header'


  def initialize( self, val=0.0 ):
    self.d = np.ones( ( self.nx, self.ny, self.nz ), np.float32 
                       ) * val
    self.data =self.d

  def initialise( self, val=0.0 ):
    self.initialize( 0.0 )

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fheader = os.path.basename( fname )

  def read_header( self, fheader=None ) :
    if fheader :
      self.fheadeer = fheader

    self.set_fname( fheader )
    
    lines = read_from_textfile( os.path.join( self.fdir, self.fheader ) ) 

    # read x-coordinate
    iline = 0 
    ox, dx, nx = lines[ iline ].split()
    self.ox = float(ox) ; self.dx = float(dx); self.nx = int(nx)
    iline += 1
    oy, dy, ny = lines[ iline ].split()
    self.oy = float(oy) ; self.dy = float(dy); self.ny = int(ny)
    iline += 1
    oz, dz, nz = lines[ iline ].split()
    self.oz = float(oz) ; self.dz = float(dz); self.nz = int(nz)
    iline += 1
    self.fbin = lines[ iline ]

    self.set_axis()
  

  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.float32 
                  ).reshape( self.nx, self.ny,  self.nz ), -9999. )
    self.data = self.d

  def read_data_fast( self ) :
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.int32 
                  ).reshape( self.nz, self.ny, self.nx ), -9999. ).astype( np.float )
    self.d = self.d.transpose( 2,1,0)
    self.data = self.d


  def read( self, fheader=None ):
    self.read_header( fheader )
    self.read_data()

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fbin = fh + '.bin' 


  def write_header( self, fheader=None, fbin=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fbin :
      self.fbin = fbin



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oy, self.dy, self.ny ) )
    outlines.append( '%f %f %d'%( self.oz, self.dz, self.nz ) )
    outlines.append( '%s'%self.fbin )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )

    try :
      self.d.filled( -9999. ).astype( np.float32 ).tofile( fbin )
    except :
      self.d.astype( np.float32 ).tofile( fbin )

  def write_data_int( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fbin )
    except :
      self.d.tofile( fbin )

  def write_data_fast( self ) :  
    fbin = os.path.join( self.fdir, self.fbin )

    self.d = self.data.astype( np.int32 )

    try :
      self.d.transpose( 2,1,0).filled( -9999 ).tofile( fbin )
    except :
      self.d.transpose(2,1,0).tofile( fbin )


  def write( self, fheader=None, fbin=None ) :
    self.write_header( fheader, fbin )
    self.write_data()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=np.float ) * self.dx + self.ox
    self.y = np.arange( 0, self.ny, dtype=np.float ) * self.dy + self.oy
    self.z = np.arange( 0, self.nz, dtype=np.float ) * self.dz + self.oz

  def norm( self ):
   self.ntrace = self.nx * self.ny * self.nz
   norm = np.dot( self.d.reshape( ( self.nx * self.nz, 1 ) )
                       [ 0:self.ntrace, 0 ], 
                   self.d.reshape( ( self.nx * self.nz, 1 ) ).T
                       [ 0, 0 : self.ntrace ] )
   norm = sqrt( norm )
   return norm
     
     
 





