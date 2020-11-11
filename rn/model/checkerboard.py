"""
  CHECKRBOARD routines
"""
import numpy as np
from rn.rsf.model import Model

def checkerboard( nx, nz, nx_checker, nz_checker ) :

  # first create slightly larger checke3rboard patterns
  nchecker_x = nx / nx_checker / 2 
  nchecker_z = nz / nz_checker / 2

  print nchecker_x, nchecker_z

  if nchecker_x * 2 <  nx :
    nchecker_x += 1 
    print nchecker_x
    ix0 = ( nchecker_x *nx_checker * 2- nx ) / 2 
  else :
    ix0 = 0 
 
  if nchecker_z *2 <  nz :
    nchecker_z += 1
    iz0 = ( nchecker_z * nz_checker * 2- nz ) / 2
  else :
    iz0 = 0


  a = np.ones( ( nx_checker, nz_checker), dtype=np.float32 )
  b = -np.ones( ( nx_checker, nz_checker ), dtype=np.float32 )

  c = np.row_stack( nchecker_x * ( a, b ) )
  d = np.row_stack( nchecker_x * ( b, a ) )

  e = np.column_stack( nchecker_z * ( c, d ))
  
  print  ix0, iz0
 
  return e[ix0:(ix0+nx), iz0:(iz0+nz)]


def model( m, x_checker, z_checker ) :
  mout = Model( ref=m )
  mout.d = checkerboard( mout.nx, mout.nz,
                            int( x_checker/mout.dx ), int( z_checker/mout.dz ) )

  return mout
