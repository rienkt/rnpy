#
# collection of classes for jogmec elastic FWI project


import rsf.api as rsf
# Import math packages
import numpy as np
import scipy as sp
import copy
from math import sqrt
from td import set_t
# define class

#--------------------------------------------------------------------------
# MCTimeDomainDataMod - Multi-component time-domain data after fdfd modelling
#                       : source - time - component - receiver 
#--------------------------------------------------------------------------
class MCTimeDomainDataMod:
  def __init__(self, nrcv=1,  nc=1,  nt=1,  nsrc=1,
                     drcv=1., dc=1., dt=1., dsrc=1.,
                     orcv=0., oc=0., ot=0., osrc=0.):
    self.nrcv=nrcv; self.nc=nc; self.nt=nt;self.nsrc=nsrc;  
    self.orcv=orcv; self.oc=oc; self.ot=ot;self.osrc=osrc;  
    self.drcv=drcv; self.dc=dc; self.dt=dt;self.dsrc=dsrc;  
    self.d=np.zeros((self.nsrc,self.nt,self.nc,self.nrcv),'f')
        
  def set_t( self ) :
    self.t = set_t( self.ot, self.dt, self.nt ) 

  def get_header( self, input ) :
    self.read_header( input=input )


  def read_header( self, input=None, f=None ) :
    if input is None :
      input = rsf.Input( f )
    self.nrcv = input.int('n1')
    self.nc   = input.int('n2')
    self.nt   = input.int('n3')
    self.nsrc = input.int('n4')
    self.drcv = input.float('d1')
    self.dc   = input.float('d2')
    self.dt   = input.float('d3')
    self.dsrc = input.float('d4')
    self.orcv = input.float('o1')
    self.oc   = input.float('o2')
    self.ot   = input.float('o3')
    self.osrc = input.float('o4')
    self.set_t()

  def write_header(self, output=None, f=None )  :
    if output is None :
      output = rsf.Output( f )
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

  def read( self, input=None, f=None ):
    if input is None :
      try :
        input = rsf.Input( f )
      except :
        print( "File %s doesn't exist"%f )
    self.read_header( input=input ) 
    self.d = np.zeros( (self.nsrc, self.nt, self.nc, self.nrcv), 'f' )
    input.read( self.d )

  def write( self, output=None, f=None ):
    if output is None :
      output = rsf.Output( output )
    self.write_header( output=output)
    output.write( self.d.astype( np.float32 ) )
    output.close()

  def zero_lag_corr(self,in2):
    zero_lag_corr = np.dot( self.d.reshape(self.nc,self.nrcv*self.nsrc*self.nt,1).T,   
                            in2.data.reshape(in2.nrcv*in2.nsrc*in2.nt,1))
    return zero_lag_corr



#------------------------------------------------------------------------------
# MCTimeDomainData - Multi-component time-domain data : src - rcv - comp - time
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
  def read_rsf(self, finput):
    try :
      input = rsf.Input(finput)
    except :
      input = finput
    self.get_header( input )
    self.initialise( 0. )
    input.read( self.d )

  def write_rsf(self,foutput):
    try :
      output=rsf.Output(foutput) 
    except :
      output=foutput
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

    self.initialize( 0 )

  def initialize( self, val=0. ) :
    self.d = np.ones( ( self.nt, self.nc, self.nx, self.nz ), 
                dtype=np.float32) * val

  def get_header(self,input):
    self.read_header( input=input )
  def read_header( self, input=None, f=None ) :
    if input is None :
      try :
        input = rsf.Input( f )
      except :
        print( "file %s does not exist"%f )
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

    self.x = np.arange( 0, self.nx, dtype=np.float ) * self.dx + self.ox
    self.z = np.arange( 0, self.nz, dtype=np.float ) * self.dz + self.oz

    self.initialize()

  def return_header(self):
    return {'nz': self.nz,'nx':self.nx,'nc':self.nc,'nt':self.nt,
        'oz': self.oz,'ox':self.ox,'oc':self.oc,'ot':self.ot,
        'dz': self.dz,'dx':self.dx,'dc':self.dc,'dt':self.dt}
  def read( self, input=None, f=None ):
    if input is None :
      try :
        input = rsf.Input( f )
      except :
       print( "file %s does not exist"%f )
    self.read_header( input=input )
    
    input.read( self.d )
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

