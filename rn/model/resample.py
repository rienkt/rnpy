#!/usr/bin/python env
#from scipy     import signal
from scipy import interpolate
from rk.rsf.model import Model

def resample1d( model, dzout ) :
  nz = model.nz
  dz = model.dz 

  # to make sure dzout * ( nzout - 1 ) < dz * ( nz - 1 )  
  nzout = int( ( nz - 1 ) * dz / dzout ) + 1
  
  modelout = Model( ox = model.ox, dx = model.dx , nx = model.nx, 
                    oz = model.oz, dz = dzout,     nz = nzout )

  modelout.set_axis()


  f = interpolate.interp1d( model.z, model.d, axis=-1 ) 


  modelout.d = f( modelout.z )


  return modelout
  


 
def resample2d( model, dxout, dzout ) :
  nz = model.nz
  dz = model.dz
  nx = model.nx
  dx = model.dx

  print( nz,dz,dzout )
 
  nzout = int( ( nz - 1 ) * dz / dzout ) + 1
  nxout = int( ( nx - 1 ) * dx / dxout ) + 1

  modelout = Model( ox = model.ox, dx = dxout, nx = nxout,
                    oz = model.oz, dz = dzout, nz = nzout )

  modelout.set_axis()

  print( model.z.shape, model.x.shape, model.d.shape )

  f = interpolate.interp2d( model.z, model.x, model.d, kind = 'linear' )
  modelout.d = f( modelout.z, modelout.x )

  return modelout
