#!/usr/bin/python env
import numpy as np
from scipy import interpolate
from .core import Model

def resample1d( model, dzout ) :
  nz = model.nz
  dz = model.dz 

  # to make sure dzout * ( nzout - 1 ) < dz * ( nz - 1 )  
  nzout = int( ( nz - 1 ) * dz / dzout ) + 1
  
  modelout = Model( ox = model.ox, dx = model.dx , nx = model.nx, 
                    oz = model.oz, dz = dzout,     nz = nzout )

  modelout.set_axis()


  f = interpolate.interp1d( model.z, model.data, axis=-1 ) 


  modelout.data = f( modelout.z )


  return modelout
  


 
def resample2d( model, dxout, dzout ) :
  nz = model.nz
  dz = model.dz
  nx = model.nx
  dx = model.dx

  print nz,dz,dzout

  nzout = int( ( nz - 1 ) * dz / dzout ) + 1
  nxout = int( ( nx - 1 ) * dx / dxout ) + 1

  modelout = Model( ox = model.ox, dx = dxout, nx = nxout,
                    oz = model.oz, dz = dzout, nz = nzout )

  modelout.set_axis()

  print model.z.shape, model.x.shape, model.data.shape

  f = interpolate.interp2d( model.z, model.x, model.data, kind = 'linear' )
  modelout.data = f( modelout.z, modelout.x )

  return modelout

#-----------------------------------------------------------------------
# interpolate but do not fill masked region 
#-----------------------------------------------------------------------
def resample2d_mask( model, dxout, dzout ):
  nz = model.nz
  dz = model.dz
  nx = model.nx
  dx = model.dx


  # * create mesh of original grid
  zz, xx = np.meshgrid( model.z, model.x )
  # ** convert to masked array
  zz = np.ma.asarray( zz ) 
  xx = np.ma.asarray( xx )
  # ** specify mask based on data
  zz.mask = model.data.mask
  xx.mask = model.data.mask

  # * specify output 

  nzout = int( ( nz - 1 ) * dz / dzout ) + 1
  nxout = int( ( nx - 1 ) * dx / dxout ) + 1

  modelout = Model( ox = model.ox, dx = dxout, nx = nxout,
                    oz = model.oz, dz = dzout, nz = nzout )

  modelout.set_axis()


  # * let's interpolate

  zzout, xxout = np.meshgrid( modelout.z, modelout.x )
  # ** interpolate data
  modelout.data = interpolate.griddata( ( zz.compressed(), xx.compressed() ),
                                         model.data.compressed(),
                                         ( zzout, xxout ), method='linear' )
  # ** interpolate mask
  mask = np.ones( ( model.nx, model.nz ), dtype=np.float )
  mask = np.ma.asarray( mask )
  mask.mask = model.data.mask
  mask2 = interpolate.griddata( ( zz.compressed(), xx.compressed() ),
                                 mask.compressed(),
                                ( zzout, xxout ), method='linear' )

  print  'mask2', mask2
  mask3 = np.ma.masked_invalid( mask2 )
  print  'mask3', mask3 
  print  mask3.mask
  modelout.data = np.ma.asarray( modelout.data )
  modelout.data.mask = mask3.mask
  #modelout.data[ np.where( mask3.mask ) ] = np.nan

  print modelout.data
  print modelout.data.mask

  return modelout
