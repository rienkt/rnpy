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
#import scipy as sp
from math import sqrt

from rn.libs.misc import wprint


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
                     ox=0., dx=1., nx=1, val=0.,
                     flag_initialize = True):
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

    if flag_initialize == True :
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

  def read_data( self , vmin=-9999., ftype = np.float32) :
    #print( vmin )
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = ftype #np.float32 
                  ).reshape( self.nx, self.nz ), vmin )
    self.data =self.d

  def read_data_fast( self, f=None ) :
    fbin = os.path.join( self.fdir, self.fbin )
    if f :
      fbin = f
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = np.int32 
                  ).reshape( self.nz, self.nx ), -9999. ).T.astype( float )
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
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz

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
      try :
        self.fdir = ref.fdir
        self.fheader = ref.fheader
      except :
        print( 'no file information is given during initialization' )
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
    #print( fbin )
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
    #print( self.fbin )


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
    #print( outlines )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )
    print( self.d[100,100] )
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
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.y = np.arange( 0, self.ny, dtype=float ) * self.dy + self.oy


  def extract( self, ix0, ix1, iy0, iy1 ) :
    m = Modelxy( dx=self.dx, dy=self.dy, nx=ix1-ix0, ny=iy1-iy0,
                ox=self.x[ix0], oy=self.y[iy0] )
    m.d = self.d[ ix0:ix1, iy0:iy1 ]
    m.ox = self.x[ix0]
    m.oy = self.y[iy0]
    return m
     
#}}}}}     


class Model3d:
#{{{{{
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
                  ).reshape( self.nz, self.ny, self.nx ), -9999. ).astype( float )
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
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.y = np.arange( 0, self.ny, dtype=float ) * self.dy + self.oy
    self.z = np.arange( 0, self.nz, dtype=float ) * self.dz + self.oz

  def norm( self ):
   self.ntrace = self.nx * self.ny * self.nz
   norm = np.dot( self.d.reshape( ( self.nx * self.nz, 1 ) )
                       [ 0:self.ntrace, 0 ], 
                   self.d.reshape( ( self.nx * self.nz, 1 ) ).T
                       [ 0, 0 : self.ntrace ] )
   norm = sqrt( norm )
   return norm
     
#}}}}}

class Modelxyt:
#{{{{{
  def __init__(self, ref=None, 
                     ot=0., dt=1., nt=1,
                     oy=0., dy=1., ny=1,
                     ox=0., dx=1., nx=1, val=0.,
                     flag_initialize = True ):
    if ref :
      self.nx = ref.nx
      self.ox = ref.ox
      self.dx = ref.dx
      self.nt = ref.nt
      self.ot = ref.ot
      self.dt = ref.dt
      self.ny = ref.ny
      self.oy = ref.oy
      self.dy = ref.dy
      self.fdir = ref.fdir
      self.fheader = ref.fheader
    else :

      self.nx = nx
      self.ox = ox
      self.dx = dx
      self.nt = nt
      self.ot = ot
      self.dt = dt
      self.ny = ny
      self.oy = oy
      self.dy = dy
    if flag_initialize :
      self.initialize( val=val )

    self.set_axis()


    self.fbin    = 'test.bin'
    self.fheader = 'test.header'


  def initialize( self, val=0.0 ):
    self.d = np.ones( ( self.nx, self.ny, self.nt ), np.float32 
                       ) * val
    self.data =self.d

  def initialise( self, val=0.0 ):
    self.initialize( 0.0 )

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fheader = os.path.basename( fname )

  def read_header( self, fheader=None, dtype=np.float32 ) :
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
    ot, dt, nt = lines[ iline ].split()
    self.ot = float(ot) ; self.dt = float(dt); self.nt = int(nt)

    iline += 1
    self.fbin = lines[ iline ]

    self.set_axis()

    self.dtype = dtype
  

  def open_data( self, op='r' ): #{{{{{
    self.fbinh = open( os.path.join( self.fdir, self.fbin ), op+'b' )

  def close_data( self ) :
    self.fbinh.close()


  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fbin = os.path.join( self.fdir, self.fbin )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = self.dtype 
                  ).reshape( self.nx, self.ny,  self.nt ), -9999. )
    self.data = self.d

  def read_data_at_x( self, x ) :
    fbin = os.path.join( self.fdir, self.fbin )

    import rn.libs.array as rn_array
    x, ix = rn_array.find_nearest_value( self.x, x )
    ibyte  = np.dtype( self.dtype ).itemsize

    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fbin, dtype = self.dtype,
                    count=self.ny*self.nt, 
                    offset=self.ny*self.nt*ix*ibyte
                  ).reshape( self.ny,  self.nt ), -9999. )

  def read_data_at_y( self, y ) :
    fbin = os.path.join( self.fdir, self.fbin )

    import rn.libs.array as rn_array
    y, iy = rn_array.find_nearest_value( self.y, y )
    ibyte  = np.dtype( self.dtype ).itemsize
    self.d = np.zeros( self.nx, self.nt, dtype=self.dtype )
    if not self.fbinh :
      self.open_data( op='r' )
    self.fbinh.seek( ibyte * iy * self.nt, os.SEEK_CUR )
    for ix in range( self.nx ) :
      self.d[ ix, : ] = np.fromfile( self.fbinh, dtype=self.dtype,
                            count = self.nt )
      self.fbinh.seek( ibyte *  self.ny * self.nt, os.SEEK_CUR )

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
    outlines.append( '%f %f %d'%( self.ot, self.dt, self.nt ) )
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



  def write( self, fheader=None, fbin=None ) :
    self.write_header( fheader, fbin )
    self.write_data()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=float ) * self.dx + self.ox
    self.y = np.arange( 0, self.ny, dtype=float ) * self.dy + self.oy
    self.t = np.arange( 0, self.nt, dtype=float ) * self.dt + self.ot

  def norm( self ):
   self.ntrace = self.nx * self.ny * self.nt
   norm = np.dot( self.d.reshape( ( self.nx * self.nt, 1 ) )
                       [ 0:self.ntrace, 0 ], 
                   self.d.reshape( ( self.nx * self.nt, 1 ) ).T
                       [ 0, 0 : self.ntrace ] )
   norm = sqrt( norm )
   return norm
     
#}}}}}
