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
  return np.arange( 0, nt , dtype=np.float) * dt + ot




# set loc class by reading rsf file

def rsf_loc( floc ) :
  input = rsf.Input( floc )
  n1 = input.int( 'n1' )
  n2 = input.int( 'n2' )
  loc = rn_loc( n=n2 ) 
  tmp = np.zeros( ( loc.n, 2 ), dtype=np.float32) 
  input.read( tmp )
  loc.x = tmp[ :, 0 ]
  loc.z = tmp[ :, 1 ]
  return loc


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
    self.set_t()
    self.initialize()

  def initialize( self, val=0. ) :
    self.d = val * np.ones((self.nsrc, self.nt, self.nrcv),'f')
    self.axis = ('s','t','r')
  def read_header( self, input=None, f=None, fsrc=None, frcv=None ):
    if input is None :
      input = rsf.Input( f )
    self.nrcv = input.int('n1')
    self.nt   = input.int('n2')
    self.nsrc = input.int('n3')
    self.drcv = input.float('d1')
    self.dt   = input.float('d2')
    self.dsrc = input.float('d3')
    self.orcv = input.float('o1')
    self.ot   = input.float('o2')
    self.osrc = input.float('o3')
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
    self.d=np.zeros((self.nsrc,self.nt,self.nrcv),'f')
    input.read(self.d)
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
    self.nsrc=input.int('n4')
    self.drcv=input.float('d1')
    self.dc=input.float('d2')
    self.dt=input.float('d3')
    self.dsrc=input.float('d4')
    self.orcv=input.float('o1')
    self.oc=input.float('o2')
    self.ot=input.float('o3')
    self.osrc=input.float('o4')
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
