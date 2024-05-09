#
# Timedomain classes

"""

  TD.PY

  Define Time-domain wavefield classes

  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#==============================================================================
#                                                                      MODULES
#==============================================================================

import rsf.api as rsf
import numpy as np
import scipy as sp
import copy
from math import sqrt


from rn.bin.active.core_nocsv import rn_loc, rn_fbreak
import rn.rsf.fd as rn_fd

from rn.rsf.utils import get_dim

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#=============================================================================
# define useful functions
def set_t( ot, dt, nt ) :
  return np.arange( 0, nt , dtype=float) * dt + ot




# set loc class by reading rsf file

class rsf_loc( ) :
  def __init__( self, n=1 ) :
    self.n = n
    #self.x = np.zeros( self.n, dtype=float )
    #self.z = np.zeros( self.n, dtype=float )
  def read( self,  floc ) :
    input = rsf.Input( floc )
    n1 = input.int( 'n1' )
    n2 = input.int( 'n2' )
    loc = rn_loc( n=n2 ) 
    tmp = np.zeros( ( loc.n, 2 ), dtype=np.float32) 
    input.read( tmp )
    loc.x = tmp[ :, 0 ]
    loc.z = tmp[ :, 1 ]
    #return loc


#--------------------------------------------------------------------------
# TimeDomainDataMod - Time domain d after fdfd modelling
#--------------------------------------------------------------------------
class TimeDomainDataMod:
  def __init__(self, ref=None, 
                     nt=1, nrcv=1, nsrc=1, dt=1., drcv=1., dsrc=1.,
                     ot=0.,orcv=0.,osrc=0.):
    if ref is not None :
      self.nt=ref.nt; self.nrcv=ref.nrcv; self.nsrc=ref.nsrc;
      self.ot=ref.ot; self.orcv=ref.orcv; self.osrc=ref.osrc;
      self.dt=ref.dt; self.drcv=ref.drcv; self.dsrc=ref.dsrc;
    else :
      self.nt=nt; self.nrcv=nrcv; self.nsrc=nsrc;
      self.ot=ot; self.orcv=orcv; self.osrc=osrc;
      self.dt=dt; self.drcv=drcv; self.dsrc=dsrc;
    self.ntrace = self.nrcv * self.nsrc
    self.ndim = 3
    self.set_t()
    self.initialize()

  def initialize( self, val=0. ) :
    if self.ndim == 3 :
      self.d = val * np.ones((self.nsrc, self.nt, self.nrcv),'f')
      self.axis = ('s','t','r')
    else :
      self.d = val * np.ones((self.nt, self.nrcv),'f')
      self.axis = ('t','r')

  def read_header( self, input=None, f=None, fsrc=None, frcv=None ):
    if input is None :
      input = rsf.Input( f )
    self.ndim =  get_dim( input )
    if self.ndim == 3 :
      self.nrcv = input.int('n1')
      self.nt   = input.int('n2')
      self.nsrc = input.int('n3')
      self.drcv = input.float('d1')
      self.dt   = input.float('d2')
      self.dsrc = input.float('d3')
      self.orcv = input.float('o1')
      self.ot   = input.float('o2')
      self.osrc = input.float('o3')
    else :
      self.nrcv = input.int('n1')
      self.nt   = input.int('n2')
      self.nsrc = 1
      self.drcv = input.float('d1')
      self.dt   = input.float('d2')
      self.dsrc = 1
      self.orcv = input.float('o1')
      self.ot   = input.float('o2')
      self.osrc = 0.
    self.t = set_t( self.ot, self.dt, self.nt )
    # setup srcs and rcvs
    if fsrc :
      self.srcs = rsf_loc( fsrc )
    else :
      self.srcs = rn_loc( self.nsrc )

    if frcv :
      self.rcvs = rsf_loc( fsrc )
    else :
      self.rcvs = rn_loc( self.nrcv )

  def return_header(self):
    return {'nsrc': self.nsrc,'nrcv':self.nrcv,'nt':self.nt,
        'osrc': self.osrc,'or':self.orcv,'ot':self.ot,
        'dsrc': self.dsrc,'drcv':self.drcv,'dt':self.dt}

  def write_header(self,output):
    output.put('n1',self.nrcv)
    output.put('n2',self.nt)
    output.put('n3',self.nsrc)
    output.put('d1',self.drcv)
    output.put('d2',self.dt)
    output.put('d3',self.dsrc)
    output.put('o1',self.orcv)
    output.put('o2',self.ot)
    output.put('o3',self.osrc)
  def write_header_only( self, f=None, fbin=None ) :
    with open( f, 'w' ) as frsf :
      frsf.write( 'data_format="native_float"\n' )
      frsf.write( 'n1=%d\n'%self.nrcv  )
      frsf.write( 'n2=%d\n'%self.nt )
      frsf.write( 'n3=%d\n'%self.nsrc )
      frsf.write( 'o1=%f\n'%self.orcv )
      frsf.write( 'o2=%f\n'%self.ot )
      frsf.write( 'o3=%f\n'%self.osrc )
      frsf.write( 'd1=%f\n'%self.drcv )
      frsf.write( 'd2=%f\n'%self.dt )
      frsf.write( 'd3=%f\n'%self.dsrc )
      frsf.write( 'in=%s\n'%fbin )

  def set_t( self ) :
    self.t = set_t( self.ot, self.dt, self.nt ) 

  def extract_time( self, tmin, tmax ) :
   
    self.set_t()
 
    itmin = np.argmin( np.abs( self.t - tmin ) )
    itmax = np.argmin( np.abs( self.t - tmax) )


    self.d = self.d[ :, itmin:(itmax+1), : ]

    self.ot = self.t[itmin]
    self.nt = itmax - itmin + 1
    self.set_t()


  def read_rsf( self, input=None, f=None ) :

    self.read( input=input, f=f )
  def read(self, input=None, f=None, fsrc=None, frcv=None):
    if input is None :
      input = rsf.Input( f )
    self.read_header( input, fsrc=fsrc, frcv=frcv )
    self.initialize()

    input.read(self.d)
#    self.d = np.zeros( (self.nsrc,self.nt,self.nrcv), np.float32 )
#    input.read( self.d ) 
    input.close()

  def write_rsf(self,output=None, f=None):
    self.write( output=output, f=f )
  def write( self, output=None, f=None):
    #print('hehe')
    if output is None :
      output = rsf.Output( f )
    #print('hehe 2')
    self.write_header(output)
    #print('hehe 3')
    output.write(np.float32(self.d))
    output.close()
  def zero_lag_corr( self, in2 ):
    zero_lag_corr = np.dot( 
                      self.d.reshape( self.nrcv * self.nsrc * self.nt, 1 ).T,
                      in2.d.reshape(  in2.nrcv * in2.nsrc * in2.nt, 1 ))
    return zero_lag_corr
#--------------------------------------------------------------------------
# TimeDomainData - Time domain d: source - receiver - time
#--------------------------------------------------------------------------
class TimeDomainData:
  def __init__(self, ref=None, 
                     nt=1, nrcv=1, nsrc=1, dt=1., drcv=1., dsrc=1.,
                     ot=0.,orcv=0.,osrc=0.):
    if ref is not None :
      self.nt=ref.nt; self.nrcv=ref.nrcv; self.nsrc=ref.nsrc;
      self.ot=ref.ot; self.orcv=ref.orcv; self.osrc=ref.osrc;
      self.dt=ref.dt; self.drcv=ref.drcv; self.dsrc=ref.dsrc;
    else :
      self.nt=nt; self.nrcv=nrcv; self.nsrc=nsrc;
      self.ot=ot; self.orcv=orcv; self.osrc=osrc;
      self.dt=dt; self.drcv=drcv; self.dsrc=dsrc;
    self.ntrace = self.nrcv * self.nsrc
    self.initialize()
  def initialize( self, val=0. ) :
    self.d = val * np.ones((self.nsrc, self.nrcv, self.nt),'f')
    self.axis = ('s','r','t')
  def initialise( self, val=0. ) :
     self.initialize( val ) 
  def read_header( self, input, fsrc=None, frcv=None ):
    self.nt   = input.int('n1')
    self.dt   = input.float('d1')
    self.ot   = input.float('o1')

    ndim = get_dim( input )

    if ndim == 1 :
      self.nrcv = 1
      self.nsrc = 1
      self.drcv = 1.
      self.dsrc = 1.
      self.orcv = 0.
      self.osrc = 0.
    elif ndim == 2 :
      self.nsrc = 1
      self.dsrc = 1.
      self.osrc = 0.
      self.nrcv = input.int('n2')
      self.drcv = input.float('d2')
      self.orcv = input.float('o2')
    else :
      self.nrcv = input.int('n2')
      self.nsrc = input.int('n3')
      self.drcv = input.float('d2')
      self.dsrc = input.float('d3')
      self.orcv = input.float('o2')
      self.osrc = input.float('o3')

    # setup srcs and rcvs
    if fsrc :
      if fsrc[-3:] == 'rsf' :
        self.srcs = rsf_loc( fsrc )
      else :
        self.srcs = rn_fd.rn_loc()
        self.srcs.read( fsrc )
    else :
      self.srcs = rn_loc( self.nsrc )

    if frcv :
      if fsrc[-3:] == 'rsf' :
        self.rcvs = rsf_loc( fsrc )
      else :
        self.rcvs = rn_fd.rn_loc()
        self.rcvs.read( frcv )
    else :
      self.rcvs = rn_loc( self.nrcv )

    self.ntrace=self.nrcv*self.nsrc
    self.t = set_t( self.ot, self.dt, self.nt )

  def set_t( self ) :
    self.t = set_t( self.ot, self.dt, self.nt ) 

  def extract_time( self, tmin, tmax ) :
   
    self.set_t()
 
    itmin = np.argmin( np.abs( self.t - tmin ) )
    itmax = np.argmin( np.abs( self.t - tmax) )


    self.d = self.d[ :, : , itmin:(itmax+1) ]

    self.ot = self.t[itmin]
    self.nt = itmax - itmin + 1
    self.set_t()



  def return_header(self):
    return {'nsrc': self.nsrc,'nrcv':self.nrcv,'nt':self.nt,
        'osrc': self.osrc,'or':self.orcv,'ot':self.ot,
        'dsrc': self.dsrc,'drcv':self.drcv,'dt':self.dt}

  def read_rsf( self, input=None, f=None ):
    self.read( input=input, f=f )

  def read( self, input=None, f=None, fbin=None, fsrc=None, frcv=None ):
    if input is None :
      input = rsf.Input( f )
    self.read_header( input=input, fsrc=fsrc, frcv=frcv)
    self.initialise()
    input.read(self.d)
    #input.close()
  def write(self, output=None, f=None):
    if output is None :
      output = rsf.Output( f )
    output.put('n1',self.nt)
    output.put('n2',self.nrcv)
    output.put('n3',self.nsrc)
    output.put('d1',self.dt)
    output.put('d2',self.drcv)
    output.put('d3',self.dsrc)
    output.put('o1',self.ot)
    output.put('o2',self.orcv)
    output.put('o3',self.osrc)
    output.write(np.float32(self.d))
    output.close()
  def write_header_only( self, f=None, fbin=None ) :
    print( self.dt )
    with open( f, 'w' ) as frsf :
      frsf.write( 'data_format="native_float"\n' )
      frsf.write( 'n1=%d\n'%self.nt )
      frsf.write( 'n2=%d\n'%self.nrcv )
      frsf.write( 'n3=%d\n'%self.nsrc )
      frsf.write( 'o1=%f\n'%self.ot )
      frsf.write( 'o2=%f\n'%self.orcv )
      frsf.write( 'o3=%f\n'%self.osrc )
      frsf.write( 'd1=%f\n'%self.dt )
      frsf.write( 'd2=%f\n'%self.drcv )
      frsf.write( 'd3=%f\n'%self.dsrc )
      frsf.write( 'in=%s\n'%fbin )
  def get_rms(self):
    self.rms=np.sqrt(np.sum(self.d**2)/self.d.size)
  def srt2nt(self):
    self.d=self.d.reshape(self.ntrace,self.nt)
    self.axis=('n','t')
  def zero_lag_corr(self,in2):
    zero_lag_corr=np.dot(self.d.reshape(self.nrcv*self.nsrc*self.nt,1).T,in2.d.     reshape(in2.nrcv*in2.nsrc*in2.nt,1))
    return zero_lag_corr

#------------------------------------------------------------------------------
# TimeDomainWavefield - time-domain wavefield (snapshot)
#                      : time - x - z
#-----------------------------------------------------------------------------
class TimeDomainWavefield:
  def __init__(self,ntrace=1,nc=1,nt=1,dtrace=1.,dc=1.,dt=1.,otrace=0.,oc=0.,ot=0.):
    self.ntrace=ntrace
    self.nc=nc
    self.nt=nt
    self.dtrace=dtrace
    self.dc=dc
    self.dt=dt
    self.otrace=otrace
    self.oc=oc
    self.ot=ot
    self.d=np.zeros((self.nt,self.nc,self.ntrace),'f')
    
  def read_header(self,input):
    self.ntrace=input.int('n1')
    self.nc=input.int('n2')
    self.nt=input.int('n3')
    self.dtrace=input.float('d1')
    self.dc=input.float('d2')
    self.dt=input.float('d3')
    self.otrace=input.float('o1')
    self.oc=input.float('o2')
    self.ot=input.float('o3')
    # we assume nt>1
    if self.nt==1:
      self.nt=self.nc
      self.nc=1
      self.dt=self.dc
      self.dc=1.
      self.ot=self.oc
      self.oc=0.

  def write_header(self,output):
    output.put('n1',self.ntrace)
    output.put('n2',self.nc)
    output.put('n3',self.nt)
    output.put('d1',self.dtrace)
    output.put('d2',self.dc)
    output.put('d3',self.dt)
    output.put('o1',self.otrace)
    output.put('o2',self.oc)
    output.put('o3',self.ot)

  def read_rsf(self,input):
    self.read_header(input)
    self.d=np.zeros((self.nt,self.nc,self.ntrace),'f')
    input.read(self.d)
  def write_rsf(self,output):
    self.write_header(output)
    output.write(np.float32(self.d))

  def read_rsf_pos(self,input):
    self.d=np.zeros((self.nc,self.ntrace),'f')
    input.read(self.d)
  def write_rsf_pos(self,output):
    output.write(np.float32(self.d))    


#--------------------------------------------------------------------------
# MCTimeDomainDataMod - Multi-component time-domain d after fdfd modelling
#                       : source - time - component - receiver 
#--------------------------------------------------------------------------
class MCTimeDomainDataMod:
  def __init__(self, ref=None, nrcv=1,  nc=1,  nt=1,  nsrc=1,
                     drcv=1., dc=1., dt=1., dsrc=1.,
                     orcv=0., oc=0., ot=0., osrc=0.):
    # needs to clean up here. 
    if ref :
      self.d = copy.copy( self.d )
    else : 
      self.nrcv=nrcv; self.nc=nc; self.nt=nt;self.nsrc=nsrc;  
      self.orcv=orcv; self.oc=oc; self.ot=ot;self.osrc=osrc;  
      self.drcv=drcv; self.dc=dc; self.dt=dt;self.dsrc=dsrc;  
      self.d=np.zeros((self.nsrc,self.nt,self.nc,self.nrcv),'f')

  #def get_header(self,input, fsrc=None, frcv=None ):
  def read_header( self, input=None, f=None, fsrc=None, frcv=None ):
    if input is None :
      input = rsf.Input( f )

    self.nrcv=input.int('n1')
    self.nc=input.int('n2')
    self.nt  =input.int('n3')
    self.drcv=input.float('d1')
    self.dc=input.float('d2')
    self.dt=input.float('d3')
    self.orcv=input.float('o1')
    self.oc=input.float('o2')
    self.ot=input.float('o3')
    if get_dim( input ) >= 4 :
      self.osrc=input.float('o4')
      self.nsrc=input.int('n4')
      self.dsrc=input.float('d4')
    else :
      self.nsrc = 1
      self.osrc = 0.
      self.dsrc =1.
    self.set_t()
    # setup srcs and rcvs
    if fsrc :
      self.srcs = rsf_loc( fsrc )
    else :
      self.srcs = rn_loc( self.nsrc )

    if frcv :
      self.rcvs = rsf_loc( fsrc )
    else :
      self.rcvs = rn_loc( self.nrcv )


  def set_t( self ) :
    self.t = self.ot + np.arange( self.nt, dtype=np.float ) * self.dt
  
  def write_header(self,output):
    output.put('n1',self.nrcv)
    output.put('n2',self.nc)
    output.put('n3',self.nt)
    output.put('n4',self.nsrc)
    output.put('d1',self.drcv)
    output.put('d2',self.dc)
    output.put('d3',self.dt)
    output.put('d4',self.dsrc)
    output.put('o1',self.orcv)
    output.put('o2',self.oc)
    output.put('o3',self.ot)
    output.put('o4',self.osrc)
  def read_rsf( self, input=None, f=None):
    self.read( input=input, f=f )

  def read( self, input=None, f=None):
    if input is None :
      input = rsf.Input( f )
    self.read_header( input=input )

    print( self.nsrc, self.nt, self.nc, self.nrcv )

    self.d = np.zeros( (self.nsrc, self.nt, self.nc, self.nrcv ),'f')
    input.read(self.d)
  def write_rsf( self, output=None, f=None ):
    if output is None :
      output = rsf.Output( f )
    self.write_header(output)
    output.write(np.float32(self.d))
    output.close()
  def zero_lag_corr(self,in2):
    zero_lag_corr = np.dot( self.d.reshape(self.nc,self.nrcv*self.nsrc*self.nt,1).T,   
                            in2.d.reshape(in2.nrcv*in2.nsrc*in2.nt,1))
    return zero_lag_corr


#--------------------------------------------------------------------------
# First break  # hahaha? 
#--------------------------------------------------------------------------
class Fbreak:
  def __init__( self, ref=None, nsrc=1, osrc=0., dsrc=1.,
                                nrcv=1, orcv=0., drcv=1. )  :
    if ref :
      self.nsrc = ref.nsrc
      self.nrcv = ref.nrcv
      self.osrc = ref.osrc
      self.orcv = ref.orcv
      self.dsrc = ref.dsrc
      self.drcv = ref.drcv
    else :
      self.nsrc = nsrc
      self.nrcv = nrcv
      self.osrc = osrc
      self.orcv = orcv
      self.dsrc = dsrc
      self.drcv = drcv
    self.initialize()
  def initialize( self, val=0. ) :
    self.d = val * np.ones((self.nsrc, self.nrcv),'f')

  def read_header(self, input=None, f=None ):
    if input is None :
      input = rsf.Input( f )
    self.nrcv = input.int('n1')
    self.nsrc = input.int('n2')
    self.drcv = input.float('d1')
    self.dsrc = input.float('d2')
    self.orcv = input.float('o1')
    self.osrc = input.float('o2')


  def write_header(self,output):
    output.put('n1',self.nrcv)
    output.put('n2',self.nsrc)
    output.put('d1',self.drcv)
    output.put('d2',self.dsrc)
    output.put('o1',self.orcv)
    output.put('o2',self.osrc)

  def read_rsf( self, input=None, f=None ) :

    self.read( input=input, f=f )

  def read(self, input=None, f=None):
    if input is None :
      input = rsf.Input( f )
    self.read_header(input)
    self.d=np.zeros((self.nsrc,self.nrcv),'f')
    input.read(self.d)

    self.d = np.ma.masked_equal( self.d, 0. )
    if type( self.d.mask ) is not np.ndarray :
      if self.d.mask is False :
        self.d.mask = np.zeros( 0, dtype=np.bool )
      else :
        self.d.mask = np.zeros( 1, dtype=np.bool )
      


  def write(self,output=None, f=None):
    if output is None :
      output = rsf.Output( f )
    self.write_header(output)
    output.write(np.float32(self.d))
    output.close()

#------------------------------------------------------------------------------
# MCTimeDomainData - Multi-component time-domain d : src - rcv - comp - time
#-----------------------------------------------------------------------------
class MCTimeDomainData:
  def __init__(self, ref=None, 
                     nt=1, nc=1, nrcv=1, nsrc=1,
                     dt=1.,dc=1.,drcv=1.,dsrc=1.,
                     ot=0.,oc=0.,orcv=0.,osrc=0.):
    if ref is not None :
      self.nt = ref.nt; self.nc = ref.nc; self.nrcv = ref.nrcv; self.nsrc = ref.nsrc;
      self.dt = ref.dt; self.dc = ref.dc; self.drcv = ref.drcv; self.dsrc = ref.dsrc;
      self.ot = ref.ot; self.oc = ref.oc; self.orcv = ref.orcv; self.osrc = ref.osrc;
    else :
      self.nt = nt; self.nc = nc; self.nrcv = nrcv; self.nsrc = nsrc;
      self.dt = dt; self.dc = dc; self.drcv = drcv; self.dsrc = dsrc;
      self.ot = ot; self.oc = oc; self.orcv = orcv; self.osrc = osrc;
    self.ntrace = self.nrcv * self.nsrc
    self.initialise()
    self.set_t()

  def initialize(self, val=0. ):
    self.d = np.ones(( self.nsrc, self.nrcv, self.nc, self.nt ), np.float32) * val
    self.axis = ('s', 'r', 'c', 't')
  def initialise(self, val=0. ):
    self.initialize(val)

  def get_header(self, input):
    self.nt   = input.int('n1')
    self.nc   = input.int('n2')
    self.nrcv = input.int('n3')
    self.nsrc = input.int('n4')
    self.dt   = input.float('d1')
    self.dc   = input.float('d2')
    self.drcv = input.float('d3')
    self.dsrc = input.float('d4')
    self.ot   = input.float('o1')
    self.oc   = input.float('o2')
    self.orcv = input.float('o3')
    self.osrc = input.float('o4')
    self.ntrace = self.nsrc*self.nrcv
    self.set_t()
  def read_rsf( self, input=None, f=None ) :
    self.read( input=input, f=f ) 
  def read(self, input=None, f=None):
    if input is None : 
      input = rsf.Input(f)
    self.get_header( input )
    self.initialise( 0. )
    input.read( self.d )

  def write_rsf( self, output=None, f=None ):
    self.write( output=output, f=f)
  def write(self, output=None, f=None):
    #print( f, output )
    if output is None :
      output = rsf.Output(f) 
    output.put('n1',self.nt)
    output.put('n2',self.nc)
    output.put('n3',self.nrcv)
    output.put('n4',self.nsrc)
    output.put('d1',self.dt)
    output.put('d2',self.dc)
    output.put('d3',self.drcv)
    output.put('d4',self.dsrc)
    output.put('o1',self.ot)
    output.put('o2',self.oc)
    output.put('o3',self.orcv)
    output.put('o4',self.osrc)
    output.write(np.float32(self.d))
    output.close()
  def get_rms(self):
    self.rms = np.sqrt( np.sum( self.d**2 ) / self.d.size )
  def srct2csrt(self):
    self.d = self.d.transpose(2, 0, 1, 3)
    self.axis = ('c', 's', 'r', 't')
  def csrt2srct(self):
    self.d = self.d.transpose(1, 2, 0, 3)
    self.axis = ('s', 'r', 'c', 't')
  def csrt2cnt(self):
    self.d=self.d.reshape(self.nc,self.ntrace,self.nt)
    self.axis=('c','n','t')
  def srct2nct(self):
    self.d=self.d.reshape(self.ntrace,self.nc,self.nt)
    self.axis=('n','c','t')
  def nct2srct(self):
    self.d=self.d.reshape(self.nsrc,self.nrcv,self.nc,self.nt)
    self.axis=('s','r','c','t')
  def nct2cnt(self):
    self.d=self.d.transpose(1,0,2)
    self.axis=('c','n','t')
  def cnt2nct(self):
    self.d=self.d.transpose(1,0,2)
    self.axis=('n','c','t')
  def cnt2srct(self):
    self.cnt2nct(); self.nct2srct();
    self.axis=('s','r','c','t')

#------------------------------------------------------------------------------
# MCTimeDomainData - Multi-component time-domain wavefield (snapshot)
#                    : time - comp - x - z
#-----------------------------------------------------------------------------
class MCTimeDomainWavefield:
  def __init__(self,nz=1,nx=1,nc=1,nt=1,
        dz=1.,dx=1.,dc=1.,dt=1.,oz=0.,ox=0.,oc=0,ot=0.):
    self.nz=nz; self.nx=nx; self.nc=nc; self.nt=nt;  
    self.dz=dz; self.dx=dx; self.dc=dc; self.dt=dt;  
    self.oz=oz; self.ox=ox; self.oc=oc; self.ot=ot;  
    self.d=np.zeros((self.nt,self.nc,self.nx,self.nz),'f')
  def get_header(self,input):
    self.nz=input.int('n1')
    self.nx=input.int('n2')
    self.nc=input.int('n3')
    self.nt=input.int('n4')
    self.dz=input.float('d1')
    self.dx=input.float('d2')
    self.dc=input.float('d3')
    self.dt=input.float('d4')
    self.oz=input.float('o1')
    self.ox=input.float('o2')
    self.oc=input.float('o3')
    self.ot=input.float('o4')
    self.ntrace=self.nx*self.nz
  def return_header(self):
    return {'nz': self.nz,'nx':self.nx,'nc':self.nc,'nt':self.nt,
        'oz': self.oz,'ox':self.ox,'oc':self.oc,'ot':self.ot,
        'dz': self.dz,'dx':self.dx,'dc':self.dc,'dt':self.dt}
  def read_rsf(self,input=None, f=None):
    if input is None :
      input = rsf.Input( f )
    self.get_header(input)
    input.read(self.d)
#    input.close()
#    self.d=self.d.transpose()
  def write_rsf(self,output):
    output.put('n1',self.nt)
    output.put('n2',self.nc)
    output.put('n3',self.nrcv)
    output.put('n4',self.nsrc)
    output.put('d1',self.dt)
    output.put('d2',self.dc)
    output.put('d3',self.drcv)
    output.put('d4',self.dsrc)
    output.put('o1',self.ot)
    output.put('o2',self.oc)
    output.put('o3',self.orcv)
    output.put('o4',self.osrc)
    output.write(np.float32(self.d))
#    output.write(self.d.transpose())
    output.close()
  def read_rsf_pos(self,input):
    self.d=np.zeros((self.nc,self.ntrace),'f')
    input.read(self.d)

#------------------------------------------------------------------------------
# TimeDomainData - time-domain wavefield (snapshot)
#                  : time - x - z
#-----------------------------------------------------------------------------
class TimeDomainWavefield:
  def __init__(self,ntrace=1,nc=1,nt=1,dtrace=1.,dc=1.,dt=1.,otrace=0.,oc=0.,ot=0.):
    self.ntrace=ntrace
    self.nc=nc
    self.nt=nt
    self.dtrace=dtrace
    self.dc=dc
    self.dt=dt
    self.otrace=otrace
    self.oc=oc
    self.ot=ot
    self.d=np.zeros((self.nt,self.nc,self.ntrace),'f')
    
  def get_header(self,input):
    self.ntrace=input.int('n1')
    self.nc=input.int('n2')
    self.nt=input.int('n3')
    self.dtrace=input.float('d1')
    self.dc=input.float('d2')
    self.dt=input.float('d3')
    self.otrace=input.float('o1')
    self.oc=input.float('o2')
    self.ot=input.float('o3')
    # we assume nt>1
    if self.nt==1:
      self.nt=self.nc
      self.nc=1
      self.dt=self.dc
      self.dc=1.
      self.ot=self.oc
      self.oc=0.

  def write_header(self,output):
    output.put('n1',self.ntrace)
    output.put('n2',self.nc)
    output.put('n3',self.nt)
    output.put('d1',self.dtrace)
    output.put('d2',self.dc)
    output.put('d3',self.dt)
    output.put('o1',self.otrace)
    output.put('o2',self.oc)
    output.put('o3',self.ot)

  def read_rsf(self,input):
    self.get_header(input)
    self.d=np.zeros((self.nt,self.nc,self.ntrace),'f')
    input.read(self.d)
  def write_rsf(self,output):
    self.write_header(output)
    output.write(np.float32(self.d))

  def read_rsf_pos(self,input):
    self.d=np.zeros((self.nc,self.ntrace),'f')
    input.read(self.d)
  def write_rsf_pos(self,output):
    output.write(np.float32(self.d))    



#------------------------------------------------------------------------------
# Steplength
#-----------------------------------------------------------------------------
class Steplen:
  def __init__(self,vel,dm,invparam,ratio=0.0):
    self.m0=vel.vel2model(invparam)
    self.dm=dm
    self.m=Model()
    self.m=copy.deepcopy(self.m0)
    self.m.initialize(0)
    self.eps=0.0
    if ratio == 0.0:
      if (invparam=='v') :
        self.ratio=0.002
      else:
        self.ratio=0.02
    else:
      self.ratio=ratio
    self.invparam=invparam

  def create_trial_model(self):
    self.eps = self.ratio*self.m0.d.max()/np.abs(self.dm.d).max()
    self.m.d=self.m0.d+self.eps*self.dm.d

  def trial_vel(self):
    vel=self.m.model2vel(self.invparam)
    return vel
    
# stiffness / compliance tensor
class SC_Tensor:
  def __init__(self, ref=None, 
                     o1=0., d1=1., n1=1,
                     o2=0., d2=1., n2=1, init_value=0.):
    if ref :
      self.n2 = ref.n2
      self.o2 = ref.o2
      self.d2 = ref.d2
      self.n1 = ref.n1
      self.o1 = ref.o1
      self.d1 = ref.d1
    else :
      self.n2 = n2
      self.o2 = o2
      self.d2 = d2
      self.n1 = n1
      self.o1 = o1
      self.d1 = d1
    
    self.n3 = 6; self.d3 = 1. ; self.o3 = 1.

    self.set_t() 
    self.initiali1e( 0.0 )

  def initiali1e( self, init_value=0.0 ):
    self.d = np.ones( ( self.n3, self.n2, self.n1 ), np.float32 
                       ) * init_value

  def initialise( self, init_value=0.0 ):
    self.initiali1e( 0.0 )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.n1 = input.int(   'n1' )
    self.n2 = input.int(   'n2' )
    self.n3 = input.int(   'n3' )
    self.d1 = input.float( 'd1' )
    self.d2 = input.float( 'd2' )
    self.d3 = input.float( 'd3' )
    self.o1 = input.float( 'o1' )
    self.o2 = input.float( 'o2' )
    self.o3 = input.float( 'o3' )

    self.nc = self.n3
    self.dc = self.d3
    self.oc = self.o3
    self.nx = self.n2
    self.dx = self.d2
    self.ox = self.o2
    self.nz = self.n1
    self.dz = self.d1
    self.oz = self.o1
    self.set_t() 
 

  def read( self, input=None, f=None ):
    if input is None :
      input = rsf.Input( f )

    self.read_header( input=input ) 

    self.initiali1e( 0.0 )
    input.read(self.d)
    self.set_axis()

  def write( self, output=None, f=None ) :
    #print('w1', f)  
    if output is None :
      output = rsf.Output( f )

    #print('w2')
    output.put( 'n1', self.n1 )
    output.put( 'n2', self.n2 )
    output.put( 'n3', self.n3 )
    output.put( 'd1', self.d1 )
    output.put( 'd2', self.d2 )
    output.put( 'd3', self.d3 )
    output.put( 'o1', self.o1 )
    output.put( 'o2', self.o2 )
    output.put( 'o3', self.o3 )
    #print('w3')
    output.write( self.d.astype( np.float32 ) )
    output.close()

  def set_axis( self ) :
    self.x = np.arange( 0, self.nx, dtype=np.float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=np.float ) * self.dz + self.oz

  def stiff2comp_iso(self):
    c11=copy.copy(self.d[0,:,:])
    c12=copy.copy(self.d[1,:,:])
    c33=copy.copy(self.d[5,:,:])
    self.d[0,]=c11/(c11**2-c12**2)
    self.d[1,]=-c12/(c11**2-c12**2)
    self.d[2,]=0.
    self.d[3,]=c11/(c11**2-c12**2)
    self.d[4,]=0.
    self.d[5,]=1./c33
  def stiff2comp_aniso(self):
    c11=copy.copy(self.d[0,:,:])
    c12=copy.copy(self.d[1,:,:])
    c33=copy.copy(self.d[5,:,:])
    self.d[0,]=c11/(c11**2-c12**2)
    self.d[1,]=-c12/(c11**2-c12**2)
    self.d[2,]=0.
    self.d[3,]=c11/(c11**2-c12**2)
    self.d[4,]=0.
    self.d[5,]=1./c33
  def vpvsden2stiff_iso(self,vp,vs,den):
    l=den.d*vp.d**2  - 2. * den.d*vs.d**2
    m=den.d*vs.d**2
    c11=l+2.*m
    c12=l
    c33=m
    self.d[0,]=c11
    self.d[1,]=c12
    self.d[2,]=0.
    self.d[3,]=c11
    self.d[4,]=0.
    self.d[5,]=c33

  def set_t( self ) :
    self.t = set_t( self.ot, self.dt, self.nt ) 




   
 
#------------------------------------------------------------------------------
# MCTimeDomainData - Multi-component time-domain d : src - rcv - comp - time
#-----------------------------------------------------------------------------
class MCTimeDomainData:
  def __init__(self, ref=None, 
                     nt=1, nc=1, nrcv=1, nsrc=1,
                     dt=1.,dc=1.,drcv=1.,dsrc=1.,
                     ot=0.,oc=0.,orcv=0.,osrc=0.):
    if ref is not None :
      self.nt = ref.nt; self.nc = ref.nc; self.nrcv = ref.nrcv; self.nsrc = ref.nsrc;
      self.dt = ref.dt; self.dc = ref.dc; self.drcv = ref.drcv; self.dsrc = ref.dsrc;
      self.ot = ref.ot; self.oc = ref.oc; self.orcv = ref.orcv; self.osrc = ref.osrc;
    else :
      self.nt = nt; self.nc = nc; self.nrcv = nrcv; self.nsrc = nsrc;
      self.dt = dt; self.dc = dc; self.drcv = drcv; self.dsrc = dsrc;
      self.ot = ot; self.oc = oc; self.orcv = orcv; self.osrc = osrc;
    self.ntrace = self.nrcv * self.nsrc
    self.initialise()
    self.set_t()

  def initialize(self, val=0. ):
    self.d = np.ones(( self.nsrc, self.nrcv, self.nc, self.nt ), np.float32) * val
    self.axis = ('s', 'r', 'c', 't')
  def initialise(self, val=0. ):
    self.initialize(val)

  def get_header(self, input):
    self.nt   = input.int('n1')
    self.nc   = input.int('n2')
    self.nrcv = input.int('n3')
    self.nsrc = input.int('n4')
    self.dt   = input.float('d1')
    self.dc   = input.float('d2')
    self.drcv = input.float('d3')
    self.dsrc = input.float('d4')
    self.ot   = input.float('o1')
    self.oc   = input.float('o2')
    self.orcv = input.float('o3')
    self.osrc = input.float('o4')
    self.ntrace = self.nsrc*self.nrcv
    self.set_t()
  def read_rsf( self, input=None, f=None ) :
    self.read( input=input, f=f ) 
  def read(self, input=None, f=None):
    if input is None : 
      input = rsf.Input(f)
    self.get_header( input )
    self.initialise( 0. )
    input.read( self.d )

  def write_rsf( self, output=None, f=None ):
    self.write( output=output, f=f)
  def write(self, output=None, f=None):
    #print( f, output )
    if output is None :
      output = rsf.Output(f) 
    output.put('n1',self.nt)
    output.put('n2',self.nc)
    output.put('n3',self.nrcv)
    output.put('n4',self.nsrc)
    output.put('d1',self.dt)
    output.put('d2',self.dc)
    output.put('d3',self.drcv)
    output.put('d4',self.dsrc)
    output.put('o1',self.ot)
    output.put('o2',self.oc)
    output.put('o3',self.orcv)
    output.put('o4',self.osrc)
    output.write(np.float32(self.d))
    output.close()
  def get_rms(self):
    self.rms = np.sqrt( np.sum( self.d**2 ) / self.d.size )
  def srct2csrt(self):
    self.d = self.d.transpose(2, 0, 1, 3)
    self.axis = ('c', 's', 'r', 't')
  def csrt2srct(self):
    self.d = self.d.transpose(1, 2, 0, 3)
    self.axis = ('s', 'r', 'c', 't')
  def csrt2cnt(self):
    self.d=self.d.reshape(self.nc,self.ntrace,self.nt)
    self.axis=('c','n','t')
  def srct2nct(self):
    self.d=self.d.reshape(self.ntrace,self.nc,self.nt)
    self.axis=('n','c','t')
  def nct2srct(self):
    self.d=self.d.reshape(self.nsrc,self.nrcv,self.nc,self.nt)
    self.axis=('s','r','c','t')
  def nct2cnt(self):
    self.d=self.d.transpose(1,0,2)
    self.axis=('c','n','t')
  def cnt2nct(self):
    self.d=self.d.transpose(1,0,2)
    self.axis=('n','c','t')
  def cnt2srct(self):
    self.cnt2nct(); self.nct2srct();
    self.axis=('s','r','c','t')
  def set_t( self ) :
    self.t = set_t( self.ot, self.dt, self.nt ) 

#--------------------------------------------------------------------------
# SourceWavelet - Source Wavelet : time - componet - source
#--------------------------------------------------------------------------
class SourceWavelet:
  def __init__(self, ref=None, 
                     nsrc=1,  nc=1,  nt=1, dsrc=1., dc=1., dt=1.,
                     osrc=0., oc=0., ot=0.):
    if ref is not None :
      self.nsrc = ref.nsrc; self.nc = ref.nc; self.nt = ref.nt; 
      self.dsrc = ref.dsrc; self.dc = ref.dc; self.dt = ref.dt; 
      self.osrc = ref.osrc; self.oc = ref.oc; self.ot = ref.ot;
    else :
      self.nsrc = nsrc; self.nc = nc; self.nt = nt; 
      self.dsrc = dsrc; self.dc = dc; self.dt = dt; 
      self.osrc = osrc; self.oc = oc; self.ot = ot;
    self.ndim = 3
    self.axis = ['t', 'c', 'r']
    self.initialise(0.)
  def get_header(self, input=None, f=None ):
    if input is None :
      input = rsf.Input( f )
    self.ndim = get_dim( input )

    if self.ndim == 3 :
      self.nsrc = input.int('n1')
      self.nc   = input.int('n2')
      self.nt   = input.int('n3')
      self.dsrc = input.float('d1')
      self.dc   = input.float('d2')
      self.dt   = input.float('d3')
      self.osrc = input.float('o1')
      self.oc   = input.float('o2')
      self.ot   = input.float('o3')
      self.set_t()
    else :
      self.nsrc = input.int('n1' )
      self.nc   = 1
      self.nt   = input.int('n2')
      self.dsrc = input.float( 'd1' )
      self.dc   = 1 
      self.dt   = input.float('d2')
      self.osrc = input.float( 'o1' )
      self.oc   = 0
      self.ot   = input.float('o2')
      self.set_t()
  def set_t( self ) :
    self.t = self.ot + np.arange( self.nt, dtype=float ) * self.dt
  def initialize(self, val=0.):
    if self.ndim == 3 :
      self.d = val * np.ones(( self.nt, self.nc, self.nsrc), dtype=np.float32)
      self.axis = ['t', 'c', 's']
    else :
      self.d = val * np.ones( ( self.nt, self.nsrc), dtype=np.float32 )
  def initialise(self, val=0.):
    self.initialize( val )
  def return_header(self):
    return {'nsrc': self.nsrc,'nc':self.nc,'nt':self.nt,
        'osrc': self.osrc,'oc':self.oc,'ot':self.ot,
        'dsrc': self.dsrc,'dc':self.dc,'dt':self.dt}
  def read_rsf( self, input=None, f=None ):
    self.read( input=input, f=f )
  def read(self,input=None, f=None):
    if input is None :
      input = rsf.Input( f )
    self.get_header(input)
    self.initialise()
    input.read(self.d)
  def write_rsf( self, output=None, f=None ):
    self.write( output=output, f=f )
  def write(self, output=None, f=None ):
    if output is None :
      output = rsf.Output( f )
    output.put('n1',self.nsrc)
    output.put('n2',self.nc)
    output.put('n3',self.nt)
    output.put('d1',self.dsrc)
    output.put('d2',self.dc)
    output.put('d3',self.dt)
    output.put('o1',self.osrc)
    output.put('o2',self.oc)
    output.put('o3',self.ot)
    output.write(np.float32(self.d))
    output.close()
  def tcs2cst(self):
    self.d=np.transpose(self.d,[1,2,0])
    self.axis=['c','s','t']  
  def cst2tcs(self):
    self.d=np.transpose(self.d,[2,0,1])


