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

# Optional dependency for HDF5 I/O
try:
  import h5py
except Exception:
  h5py = None

# Import math packages
import numpy as np
#import scipy as sp
from math import sqrt

from rn.libs.misc import wprint


#======================================================================
# HDF5 helpers
#======================================================================

def _is_hdf5(fname):
  """Return True if file extension indicates HDF5."""
  if fname is None:
    return False
  ext = os.path.splitext(str(fname))[-1].lower()
  return ext in ('.h5', '.hdf5', '.hdf')


def _require_h5py():
  if h5py is None:
    raise ImportError(
      "h5py is required for HDF5 I/O but could not be imported. "
      "Install it (e.g., pip install h5py) or write to a .bin file instead."
    )


def _h5_write(fpath, arr, fill_value=-9999., dset_name='data', attrs=None,
              compression='gzip', compression_opts=4):
  """Write a numpy array/masked array to HDF5 as float32."""
  _require_h5py()

  # Ensure directory exists
  fdir = os.path.dirname(fpath)
  if fdir and (not os.path.isdir(fdir)):
    os.makedirs(fdir, exist_ok=True)

  # Mask handling
  try:
    data = arr.filled(fill_value)
  except Exception:
    data = arr

  data = np.asarray(data, dtype=np.float32)

  with h5py.File(fpath, 'w') as hf:
    ds = hf.create_dataset(
      dset_name,
      data=data,
      dtype=np.float32,
      compression=compression,
      compression_opts=compression_opts,
      shuffle=True,
      fletcher32=True,
    )
    ds.attrs['fill_value'] = np.float32(fill_value)
    if attrs:
      for k, v in attrs.items():
        try:
          ds.attrs[k] = v
        except Exception:
          # Best-effort: skip non-HDF5-serializable attributes
          pass


def _h5_read(fpath, vmin=-9999., dset_name='data', ftype=np.float32):
  """Read an HDF5 dataset and return a masked array."""
  _require_h5py()
  with h5py.File(fpath, 'r') as hf:
    if dset_name in hf:
      ds = hf[dset_name]
    else:
      # Fall back to the first dataset
      first_key = next(iter(hf.keys()))
      ds = hf[first_key]
    data = np.asarray(ds[...], dtype=ftype)

    # Prefer dataset-stored fill value if present
    fv = ds.attrs.get('fill_value', vmin)
  return np.ma.masked_less_equal(data, fv)


#======================================================================
# classes / functions
#======================================================================

def read_from_textfile( ftxt ) :
  with open( ftxt, 'r' ) as f :
    lines = f.read().splitlines()
  return lines

def write_to_textfile( ftxt, outlines ) :
  #print( outlines )
  #print('a m')
  #print( ftxt )
  with open( ftxt, 'w' ) as f :
    f.write( '\n'.join( outlines ) )
 

class Model:
#{{{{{
  def __init__(self, ref=None, 
                     oz=0., dz=1., nz=1,
                     ox=0., dx=1., nx=1, val=0.,
                     initialize=True,
                     flag_initialize = None):
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

    if flag_initialize is not None :
      initialize = flag_initialize
    if initialize == True:
      self.initialize( val=val )

    self.set_axis()


    # Default output is now HDF5 (float32), keeping the header format.
    self.fhdf    = 'test.h5'
    self.fheader = 'test.header'
    self.fhdfh = None


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
    self.fhdf = lines[ iline ]

    self.set_axis()
  

  def read_data( self , vmin=-9999., ftype = np.float32) :
    #print( vmin )
    fhdf = os.path.join( self.fdir, self.fhdf )
    if _is_hdf5(fhdf):
      self.d = _h5_read(fhdf, vmin=vmin, ftype=ftype)
    else:
      self.d = np.ma.masked_less_equal(
                    np.fromfile( fhdf, dtype = ftype #np.float32
                    ).reshape( self.nx, self.nz ), vmin )
    self.data =self.d

  def read_data_fast( self, f=None ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    if f :
      fhdf = f
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fhdf, dtype = np.int32 
                  ).reshape( self.nz, self.nx ), -9999. ).T.astype( float )
    self.data = self.d

  def read( self, fheader=None, vmin=-9999. ):
    self.read_header( fheader )
    self.read_data(vmin=vmin)

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fhdf = fh + '.h5' 


  def write_header( self, fheader=None, fhdf=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fhdf :
      self.fhdf = fhdf



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oz, self.dz, self.nz ) )
    outlines.append( '%s'%self.fhdf )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )

    if _is_hdf5(fhdf):
      attrs = dict(
        nx=int(self.nx), nz=int(self.nz),
        ox=float(self.ox), dx=float(self.dx),
        oz=float(self.oz), dz=float(self.dz),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      # Legacy binary
      try :
        self.d.filled( -9999. ).astype( np.float32 ).tofile( fhdf )
      except :
        self.d.astype( np.float32 ).tofile( fhdf )

  def write_data_int( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fhdf )
    except :
      self.d.tofile( fhdf )

  def write_data_fast( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    #print( self.d.shape )
    self.d = self.d.astype( np.int32 )
    #print( self.d.shape )

    try :
      self.d.T.filled( -9999 ).tofile( fhdf )
    except :
      self.d.T.tofile( fhdf )


  def write( self, fheader=None, fhdf=None ) :
    self.write_header( fheader, fhdf )
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
                     ox=0., dx=1., nx=1, val=0., 
                     initialize=True,
                     flag_initialize=None,
                     ):
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


    if flag_initialize is not None :
      initialize = flag_initialize
    if flag_initialize is not None :
      initialize = flag_initialize
    if initialize :
      self.initialize( val=val )

    self.set_axis()


    # Default output is now HDF5 (float32), keeping the header format.
    self.fhdf    = 'test.h5'
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
    self.fhdf = lines[ iline ]

    self.set_axis()
  

  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    #print( fhdf )
    if _is_hdf5(fhdf):
      self.d = _h5_read(fhdf, vmin=-9999., ftype=np.float32)
    else:
      self.d = np.ma.masked_less_equal(
                    np.fromfile( fhdf, dtype = np.float32
                   ).reshape( self.nx, self.ny ), -9999. )
    self.data = self.d


  def read( self, fheader=None ):
    self.read_header( fheader )
    self.read_data()

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fhdf = fh + '.h5' 
    #print( self.fhdf )


  def write_header( self, fheader=None, fhdf=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fhdf :
      self.fhdf = fhdf

    #print( 'here' )    
    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oy, self.dy, self.ny ) )
    outlines.append( '%s'%self.fhdf )
    #print( outlines )
    #print( os.path.join( self.fdir, self.fheader ) )
    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    #print( self.d[100,100] )
    if _is_hdf5(fhdf):
      attrs = dict(
        nx=int(self.nx), ny=int(self.ny),
        ox=float(self.ox), dx=float(self.dx),
        oy=float(self.oy), dy=float(self.dy),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      # Legacy binary
      try :
        self.d.filled( -9999. ).astype( np.float32 ).tofile( fhdf )
      except :
        self.d.astype( np.float32 ).tofile( fhdf )

  def write_data_int( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fhdf )
    except :
      self.d.tofile( fhdf )

  def write_data_fast( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.data.astype( np.int32 )

    try :
      self.d.T.filled( -9999 ).tofile( fhdf )
    except :
      self.d.T.tofile( fhdf )


  def write( self, fheader=None, fhdf=None ) :
    self.write_header( fheader, fhdf )
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


    # Default output is now HDF5 (float32), keeping the header format.
    self.fhdf    = 'test.h5'
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
    self.fhdf = lines[ iline ]

    self.set_axis()
  

  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    if _is_hdf5(fhdf):
      self.d = _h5_read(fhdf, vmin=-9999., ftype=np.float32)
    else:
      self.d = np.ma.masked_less_equal(
                    np.fromfile( fhdf, dtype = np.float32
                    ).reshape( self.nx, self.ny,  self.nz ), -9999. )
    self.data = self.d

  def read_data_fast( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    self.d = np.ma.masked_less_equal( 
                  np.fromfile( fhdf, dtype = np.int32 
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
    self.fhdf = fh + '.h5' 


  def write_header( self, fheader=None, fhdf=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fhdf :
      self.fhdf = fhdf



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oy, self.dy, self.ny ) )
    outlines.append( '%f %f %d'%( self.oz, self.dz, self.nz ) )
    outlines.append( '%s'%self.fhdf )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )

    if _is_hdf5(fhdf):
      attrs = dict(
        nx=int(self.nx), ny=int(self.ny), nz=int(self.nz),
        ox=float(self.ox), dx=float(self.dx),
        oy=float(self.oy), dy=float(self.dy),
        oz=float(self.oz), dz=float(self.dz),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      # Legacy binary float32
      try :
        self.d.filled( -9999. ).astype( np.float32 ).tofile( fhdf )
      except :
        self.d.astype( np.float32 ).tofile( fhdf )

  def write_data_int( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fhdf )
    except :
      self.d.tofile( fhdf )

  def write_data_fast( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.data.astype( np.int32 )

    try :
      self.d.transpose( 2,1,0).filled( -9999 ).tofile( fhdf )
    except :
      self.d.transpose(2,1,0).tofile( fhdf )


  def write( self, fheader=None, fhdf=None ) :
    self.write_header( fheader, fhdf )
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
                     flag_initialize=None,
                     initialize = True,
                      ):
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
      if hasattr(ref, 'fdir'):
        self.fdir = ref.fdir
      #@if hasattr(ref, 'fheader'):
      #  self.fheader = ref.fheader
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
      self.fdir = ''
      #self.fheader = 'test'
    if flag_initialize is not None :
      initialize = flat_initialize
    if initialize :
      self.initialize( val=val )

    self.set_axis()

    self.fdir = ''
    # Default output is now HDF5 (float32), keeping the header format.
    self.fhdf    = 'test.h5'
    self.fheader = 'test.header'
    self.fhdfh = None


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
   
    #print( 'dir', 'bin', self.fdir, self.fheader )

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
    self.fhdf = lines[ iline ]

    self.set_axis()

    self.dtype = dtype
  

  def open_data( self, op='r' ): #{{{{{
    self.fhdfh = open( os.path.join( self.fdir, self.fhdf ), op+'b' )

  def close_data( self ) :
    self.fhdfh.close()


  def read_bin( self ) :
    self.read_data()

  def read_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    if _is_hdf5(fhdf):
      self.d = _h5_read(fhdf, vmin=-9999., ftype=self.dtype)
    else:
      self.d = np.ma.masked_less_equal(
                    np.fromfile( fhdf, dtype = self.dtype
                    ).reshape( self.nx, self.ny,  self.nt ), -9999. )
    self.data = self.d

  def read_data_at_x( self, x ) :
  
    try : 
      self.set_x_fhdf(x)
      fhdf = os.path.join( self.fdir, self.fhdf )
      print( 'read from xonly file' )
      if _is_hdf5(fhdf):
        self.d = _h5_read(fhdf, vmin=-9999., ftype=self.dtype)
      else:
        self.d = np.ma.masked_less_equal(
                      np.fromfile( fhdf, dtype = self.dtype
                      ).reshape( self.ny,  self.nt ), -9999.  )
      print( self.d )

    except :
      print( 'extract from all volume' )
      #self.set_fname( self.fheader)
      self.read_header(  )
      fhdf = os.path.join( self.fdir, self.fhdf )
      import rn.libs.array as rn_array
      x, ix = rn_array.find_nearest_value( self.x, x )
      ibyte  = np.dtype( self.dtype ).itemsize
      if _is_hdf5(fhdf):
        # Simpler but potentially memory-heavy path: read volume and slice.
        vol = _h5_read(fhdf, vmin=-9999., ftype=self.dtype)
        self.d = vol[ix, :, :]
      else:
        self.d = np.ma.masked_less_equal(
                      np.fromfile( fhdf, dtype = self.dtype,
                        count=self.ny*self.nt,
                        offset=self.ny*self.nt*ix*ibyte
                      ).reshape( self.ny,  self.nt ), -9999. )

  def read_data_at_y( self, y ) :
    try : 
      self.set_y_fhdf(y)
      fhdf = os.path.join( self.fdir, self.fhdf )
      print( fhdf )
      print( 'read from yonly file' )
      if _is_hdf5(fhdf):
        self.d = _h5_read(fhdf, vmin=-9999., ftype=self.dtype)
      else:
        self.d = np.ma.masked_less_equal(
                      np.fromfile( fhdf, dtype = self.dtype
                      ).reshape( self.nx,  self.nt ), -9999.  )
    except :
      print( 'extract from all volume' )
      self.read_header()
      fhdf = os.path.join( self.fdir, self.fhdf )

      import rn.libs.array as rn_array
      y, iy = rn_array.find_nearest_value( self.y, y )
      ibyte  = np.dtype( self.dtype ).itemsize
      self.d = np.zeros( ( self.nx, self.nt ), dtype=self.dtype )
      if _is_hdf5(fhdf):
        vol = _h5_read(fhdf, vmin=-9999., ftype=self.dtype)
        self.d = vol[:, iy, :]
      else:
        if not self.fhdfh :
          self.open_data( op='r' )
        self.fhdfh.seek( 0, 0 )
        self.fhdfh.seek( ibyte * iy * self.nt, 0 )
        for ix in range( self.nx ) :
          self.d[ ix, : ] = np.fromfile( self.fhdfh, dtype=self.dtype,
                                count = self.nt )
          self.fhdfh.seek( ibyte *  ( self.ny -1 ) * self.nt, os.SEEK_CUR )
    print( self.d.max() )

  def read( self, fheader=None ):
    self.read_header( fheader )
    self.read_data()

  def set_default_fnames( self, fhead )  :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader = fh + '.header' 
    self.fhdf = fh + '.h5' 


  def write_header( self, fheader=None, fhdf=None) :
    if fheader :
      self.fheader = fheader
      self.set_fname( fheader )
    if fhdf :
      self.fhdf = fhdf



    outlines = []
    outlines.append( '%f %f %d'%( self.ox, self.dx, self.nx ) )
    outlines.append( '%f %f %d'%( self.oy, self.dy, self.ny ) )
    outlines.append( '%f %f %d'%( self.ot, self.dt, self.nt ) )
    outlines.append( '%s'%self.fhdf )

    write_to_textfile( os.path.join( self.fdir, self.fheader ), outlines ) 

  def write_data( self ) :
    fhdf = os.path.join( self.fdir, self.fhdf )
    if _is_hdf5(fhdf):
      attrs = dict(
        nx=int(self.nx), ny=int(self.ny), nt=int(self.nt),
        ox=float(self.ox), dx=float(self.dx),
        oy=float(self.oy), dy=float(self.dy),
        ot=float(self.ot), dt=float(self.dt),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      # Legacy binary
      try :
        self.d.filled( -9999. ).astype( np.float32 ).tofile( fhdf )
      except :
        self.d.astype( np.float32 ).tofile( fhdf )

  def write_data_int( self ) :  
    fhdf = os.path.join( self.fdir, self.fhdf )

    self.d = self.d.astype( np.int32 )

    try :
      self.d.filled( -9999 ).tofile( fhdf )
    except :
      self.d.tofile( fhdf )



  def write( self, fheader=None, fhdf=None ) :
    self.write_header( fheader, fhdf )
    self.write_data()

  def set_x_fhdf( self, x ) :
    #self.read_header(  )
    if self.fhdf.lower().endswith('.h5'):
      self.fhdf = self.fhdf.replace( '.h5', '_x%.3f.h5'%x )
    elif self.fhdf.lower().endswith('.hdf5'):
      self.fhdf = self.fhdf.replace( '.hdf5', '_x%.3f.hdf5'%x )
    else:
      self.fhdf = self.fhdf.replace( '.bin', '_x%.3f.bin'%x )
  def set_y_fhdf( self, y ) :
    #self.read_header(  )
    if self.fhdf.lower().endswith('.h5'):
      self.fhdf = self.fhdf.replace( '.h5', '_y%.3f.h5'%y )
    elif self.fhdf.lower().endswith('.hdf5'):
      self.fhdf = self.fhdf.replace( '.hdf5', '_y%.3f.hdf5'%y )
    else:
      self.fhdf = self.fhdf.replace( '.bin', '_y%.3f.bin'%y )

  def write_data_at_x( self, x,  fhdf=None ) :
    
    if fhdf :
      self.fhdf = fhdf
    else :
      self.set_x_fhdf(x) 
    fhdf = os.path.join( self.fdir, self.fhdf ) 
    if _is_hdf5(fhdf):
      attrs = dict(
        ny=int(self.ny), nt=int(self.nt),
        oy=float(self.oy), dy=float(self.dy),
        ot=float(self.ot), dt=float(self.dt),
        x=float(x),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      try :
        self.d.filled( -9999 ).tofile( fhdf )
      except :
        self.d.tofile( fhdf )

  def write_data_at_y( self, y,  fhdf=None ) :
    
    if fhdf :
      self.fhdf = fhdf
    else :
      self.set_y_fhdf(y) 
    fhdf = os.path.join( self.fdir, self.fhdf ) 
#    print( 'writing', self.d.max() )
    if _is_hdf5(fhdf):
      attrs = dict(
        nx=int(self.nx), nt=int(self.nt),
        ox=float(self.ox), dx=float(self.dx),
        ot=float(self.ot), dt=float(self.dt),
        y=float(y),
      )
      _h5_write(fhdf, self.d, fill_value=-9999., attrs=attrs)
    else:
      try :
        self.d.filled( -9999 ).tofile( fhdf )
      except :
        self.d.tofile( fhdf )
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
