#!/usr/bin/env python
from __future__ import print_function
import datetime
import numpy as np
import pytz 
import os 
import copy
try :
  import utm
except :
  print( "no utm found" )
import pandas as pd
from collections import OrderedDict


tz_jp = pytz.timezone('Asia/Tokyo')


class rn_loc( object ) : #{{{{{
  def __init__(self, n=1, ref=None ) : 
    #{{{{{
    if ref : 
      self.n = ref.n
      self.initialize() 
      try :
        self.db = ref.db
      except :
        print( 'Panda db has not been created' )
      #self.db = pd.DataFrame(
      #    OrderedDict( {'time': self.times.reshape( self.n ) , 
      #                  'srcid': self.srcsid.reshape( self.n ),
      #                 'rcvid': self.rcvsid.reshape( self.n )} ) )
      self.x[:] = ref.x
      self.y[:] = ref.y
      self.z[:] = ref.z
      try :
        self.lat[:] = ref.lat
        self.lon[:] = ref.lon
      except :
        self.lat[:] = ref.x
        self.lon[:] = ref.y

      try :
        sef.fname = ref.fname 
      except :
        self.fname = ''
      try  :
        self.fdir = ref.fdir
      except :
        self.fdir = ''
      try :
        self.id = ref.id
      except :
        self.id = []
        for i in range( self.n )  :
          self.id.append( '%d'%i )
      try :
        self.md = ref.md
      except :
        self.md = []

      try :
        self.time = ref.time 
      except :
        self.time = []
        for i in range( self.n ) :
          self.time.append( datetime.datetime( 1900, 1, 1, tzinfo = tz_jp ) )

    else :
      self.n  = n
      # initialize panda db
      self.fname = ''
      self.fdir  = ''
      self.initialize(  )

  #}}}}}
  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fname = os.path.basename( fname )

  def set_n( self, n=1 ) :
    self.n  = n
 
  def initialize( self, val=0.) :
    self.x = np.zeros( self.n, dtype=float )
    self.y = np.zeros( self.n, dtype=float )
    self.z = np.zeros( self.n, dtype=float )
    self.lat = np.zeros( self.n, dtype=float )
    self.lon = np.zeros( self.n, dtype=float )
    self.id = [] #np.arange( self.n, dtype=int )
    self.time = []
    self.dist = None
    for i in range( self.n ) :
      self.id.append( 'id-%d'%i )
      self.time.append( datetime.datetime( 1900, 1, 1, tzinfo = tz_jp ) )
    self.id = np.asarray( self.id, dtype=np.unicode_ )
  def extract( self, idxs ) :
    self.x = self.x[idxs]
    self.y = self.y[idxs]
    self.z = self.z[idxs]
    self.lat = self.lat[idxs]
    self.lon = self.lon[idxs]
    self.id = self.id[idxs]
    self.time = self.time[idxs]
    self.n = self.x.size


  def read( self, fname=None) :
    if fname :
      self.set_fname( fname )

    self.db = pd.read_csv( os.path.join( self.fdir, self.fname ) )
    #print( self.db.id )
    #print( self.db )
    self.n = len( self.db )

    self.id = self.db['id'].to_numpy()
    self.x  = self.db['x'].to_numpy()
    self.y  = self.db['y'].to_numpy()
    self.z = self.db['z'].to_numpy()
    self.lat = self.db['lat'].to_numpy()
    self.lon = self.db['lon'].to_numpy()
    self.time = self.db.time.to_numpy()

#    try :
#      self.time = [ datetime.datetime.strptime( 
#                      line.split()[4], '%Y-%m-%d-%H:%M:%S:%f',
#                      tzinfo = tz_jp )
#                      for line in lines ]
#    except :
#      self.time = None #[ datetime.datetime( 1900, 1, 1, tzinfo = tz_jp) for line in lines ]
#
  def db2value( self ) :
    self.id = self.db['id'].to_numpy()
    self.x  = self.db['x'].to_numpy()
    self.y  = self.db['y'].to_numpy()
    self.z = self.db['z'].to_numpy()
    self.lat = self.db['lat'].to_numpy()
    self.lon = self.db['lon'].to_numpy()
    self.time = self.db.time.to_numpy()
    self.dist = self.db['dist'].to_numpy()

  def write( self ) :
    self.db = pd.DataFrame(
        OrderedDict( {'x': self.x, 'y': self.y, 'z': self.z, 
                      'lat': self.lat, 'lon': self.lon,
                      'id': self.id, 'time': self.time } ) )
    if self.dist is not None :
      self.db['dist'] = self.dist
    #print( 'i am writing' )
    #print( self.db )
    self.db.to_csv( os.path.join( self.fdir, self.fname ) )

  def latlon2utm( self, lat='x' ) :
    try :
      for i in range( self.n ) :
        tmp = utm.from_latlon( self.lat[i], self.lon[i] )
        self.x[i] = tmp[0]
        self.y[i] = tmp[1]
    except :
      try :
        for i in range( self.n ) :
          tmp = utm.from_latlon( self.y[i], self.x[i] )
          self.lat[i] = self.y[i]
          self.lon[i] = self.x[i]
          self.x[i] = tmp[0]
          self.y[i] = tmp[1]
      except :
        for i in range( self.n ) :
          tmp = utm.from_latlon( self.x[i], self.y[i] )
          self.lat[i] = self.x[i]
          self.lon[i] = self.y[i]
          self.x[i] = tmp[0]
          self.y[i] = tmp[1]


#}}}}}

class rn_offset( object ) : #{{{{{
  def __init__(self, n=1 ) : 
    self.n  = n
    self.offset  = np.zeros(self.n) 
    self.fname = ''
    self.fdir  = ''
    self.id = []
    self.md = []
    for i in range( self.n )  :
        self.id.append( '%d'%i )
  
  def set_n( self, n=1 ) :
    self.n  = n
 
  def initialize( self, val=0.) :
    self.offset = np.zeros( self.n, dtype=float )
    self.id = []
    self.time = []
    for i in range( self.n ) :
      self.id.append( 'id-%d'%i )

  def read( self ) :
    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.set_n( len(lines) )

    self.id = np.array( [ line.split()[0] for line in lines ],
                        dtype=np.unicode_ )
    self.d  = np.array( [ line.split()[1] for line in lines ], 
                        dtype=float ) 


  def write( self ) :
    outlines = []
    outlines.append( '%s %f %f %f %s' % 
                       ( self.id[i], self.d[i]
                        )
                   )
 
    with open( os.path.join( self.fdir, self.fname ), 'w' ) as f :
      f.write( '\n'.join( outlines ) )

#}}}}}

class rn_fbreak( object ) : #{{{{{
  def __init__( self ) :
    self.time    = []
    self.fname = None #'test.fbreak'  
    self.n = 1

  def initialize( self, srcs, rcvs ) :
    self.time = np.ma.zeros( ( srcs.n, rcvs.n ), dtype=float )
    self.isrc = np.zeros( ( srcs.n, rcvs.n ), dtype=int)
    self.ircv = np.zeros( ( srcs.n, rcvs.n ), dtype=int)

    try :
      ( self.srcid, self.rcvid ) = np.meshgrid( srcs.id, rcvs.id )

    except :
      self.srcid = np.zeros( ( srcs.n, rcvs.n ), dtype=np.unicode_)
      self.rcvid = np.zeros( ( srcs.n, rcvs.n ), dtype=np.unicode_)
    




    self.n = srcs.n * rcvs.n
    self.db = pd.DataFrame(
        OrderedDict( {'time': self.time.reshape( self.n ) , 
                      'srcid': self.srcid.reshape( self.n ),
                      'rcvid': self.rcvid.reshape( self.n )} ) )
    

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fname = os.path.basename( fname )

  def read( self, srcs=None, rcvs=None, nsrc=0, nrcv=0,fname=None ) :
    if fname :
      self.set_fname( fname )

    self.db = pd.read_csv( os.path.join( self.fdir, self.fname ) )

    #print ( self.db )
    if nsrc == 0 :
      if srcs :
        nsrc = srcs.n
      else :
        nsrc = len( self.db.isrc.unique() )
    if nrcv == 0 :
      if rcvs :
        nrcv = rcvs.n
      else :
        nrcv = len( self.db.ircv.unique() )

    #print( nsrc, nrcv )

    try :
      self.isrc = self.db.isrc.to_numpy().reshape( nsrc, nrcv )
    except :
      self.isrc = self.db.isrcs.to_numpy().reshape( nsrc, nrcv )
    try :
      self.ircv = self.db.ircv.to_numpy().reshape( nsrc, nrcv )
    except :
      self.ircv = self.db.ircvs.to_numpy().reshape( nsrc, nrcv )


    try :
      self.srcid = self.db.srcid.to_numpy().reshape( nsrc, nrcv )
    except :
      self.srcid = self.db.srcsid.to_numpy().reshape( nsrc, nrcv )
    try :
      self.rcvid = self.db.rcvid.to_numpy().reshape( nsrc, nrcv )
    except :
      self.rcvid = self.db.rcvsid.to_numpy().reshape( nsrc, nrcv )
    try :
      self.time = np.ma.masked_equal( self.db.time.to_numpy(), 0.0 )
      self.time = self.time.reshape( nsrc, nrcv ) 
    except :
      self.time = np.ma.masked_equal( self.db.time.to_numpy(), 0.0 )
      self.time = self.time.reshape( nsrc, nrcv ) 



    #print self.srcsid, self.rcvsid, self.time
              
  
  def write( self , srcs=None, rcvs=None ) :
    #self.n = srcs.n * rcvs.n
    self.n = self.time.size

    self.db = pd.DataFrame(
        OrderedDict( {'time': self.time.reshape( self.n ).filled(0.0) , 
                      'isrc': self.isrc.reshape( self.n ),
                      'ircv': self.ircv.reshape( self.n ),
                      'srcid': self.srcid.reshape( self.n ),
                      'rcvid': self.rcvid.reshape( self.n )} ) )
    #self.db.times = self.times.filled( 0.0 ).reshape( self.n )
    
   # self.db.times = self.db.time.filled( 0.0 )
    self.db.to_csv( os.path.join( self.fdir, self.fname ) )


#}}}}}

class rn_freq( object ) : #{{{{{
  def __init__(self, n=1 ) : 
    self.n  = n
    self.d  = np.zeros(self.n)
    self.fname = ''
    self.fdir  = ''
  
  def set_n( self, n=1 ) :
    self.n  = n
 
  def initialize( self, val=0.) :
    self.d = np.ones( self.n, dtype=float ) * val

  def read( self ) :
    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.set_n( len(lines) )

    self.d = np.array( [ line.split()[0] for line in lines ],
                        dtype=float )

  def write( self ) :
    outlines = []
    for i in range(self.n) :
      outlines.append( '%f'% self.d[i] ) 

    with open( os.path.join( self.fdir, self.fname ), 'w' ) as f :
      f.write( '\n'.join( outlines ) )
#}}}}}

class binary( ) :

  def __init__( self, ref=None,  nrcv=1, nsrc=1, srcs=None, rcvs=None): #{{{{{
    if ref is None :
      if rcvs :
        self.rcvs = rn_loc( ref=rcvs )
      else :
        self.rcvs = rn_loc( n=nrcv )
      if srcs : 
        self.srcs = rn_loc( ref=srcs )
      else :
        self.srcs = rn_loc( n=nsrc )


      self.fbreak = rn_fbreak()
      self.fbreak.initialize( #n=self.srcs.n*self.rcvs.n,
           srcs=self.srcs, rcvs=self.rcvs )

      self.isrc = None
      self.ircv = None
      self.gather = 'all'
      self.sort   = 'shot'

      self.fdir    = ''
      self.fbin    = 'test.bin'
      self.fheader = 'test.header' 


      self.ircvs, self.isrcs = np.meshgrid( np.arange( nrcv, dtype=int ),
                                  np.arange( nsrc, dtype=int ) )

      self.set_cmp()


    else :

      self.rcvs = copy.deepcopy( ref.rcvs )
      self.srcs = copy.deepcopy( ref.srcs )
      self.fbreak = copy.deepcopy( ref.fbreak )

      self.fheader = ref.fheader
      self.fdir    = ref.fdir

#      self.itraces = ref.itraces
      self.isrcs   = ref.isrcs
      self.ircvs   = ref.ircvs

      self.gather = ref.gather
      self.sort   = ref.sort


    self.fbinh = None
    #}}}}}
  def read_header( self, fheader=None ) :
    if fheader is not None :
      self.fheader = fheader

    self.fdir = os.path.dirname( self.fheader )
    self.fheader = os.path.basename( self.fheader )


  def set_cmp( self ) :  #{{{{{
    if type( self.isrcs ) is np.ndarray :
      rrx = self.rcvs.x[ self.ircvs ]
      rry = self.rcvs.y[ self.ircvs ]
      rrz = self.rcvs.z[ self.ircvs ]
      ssx = self.srcs.x[ self.isrcs ]
      ssy = self.srcs.y[ self.isrcs ]
      ssz = self.srcs.z[ self.isrcs ]
    else : 
      rrz, ssz = np.meshgrid( self.rcvs.z, self.srcs.z )
      rry, ssy = np.meshgrid( self.rcvs.y, self.srcs.y )
      rrx, ssx = np.meshgrid( self.rcvs.x, self.srcs.x )
   
      self.ircvs, self.isrcs = np.meshgrid( np.arange( self.rcvs.n ),
                                        np.arange( self.srcs.n ) )

    self.cmpx = ( rrx + ssx ) * 0.5
    self.cmpy = ( rry + ssy ) * 0.5
    self.cmpz = ( rrz + ssz ) * 0.5

    self.xoffset = rrx - ssx
    self.yoffset = rry - ssy
    self.zoffset = rrz - ssz

    self.offset = np.sqrt( ( rry-ssy ) **2 +( rrx-ssx ) **2  + ( rrz-ssz ) ** 2 ) 
    self.xyangles = np.angle( ( rrx+1j*rry) -  ( ssx + 1j*ssy )  )
    self.xyangles[ self.xyangles < 0. ] += np.pi

    # dip
    self.xzangles = - np.pi/2. + np.angle( ( rrx+1j*rrz) -  ( ssx + 1j*ssz )  )
    self.xzangles = np.angle( np.exp(1j*self.xzangles) )
    #self.xzangles = np.angle( exp(1j*self.xzangles) - 1j )
    self.xzangles[ self.xzangles < 0. ] += np.pi
 #}}}}}

  def set_cmp_bin( self, ncmp, noffset, nangle ) : #{{{{{

    self.cmps   = rn_loc( ncmp )
    self.offsets = rn_loc( noffset )
    #}}}}}
  
  def set_offset2d( self, sign=0 ) : #{{{{{

    if self.isrcs.ndim > 1 :
      rrx, ssx = np.meshgrid( self.rcvs.x, self.srcs.x ) 
      rry, ssy = np.meshgrid( self.rcvs.y, self.srcs.y ) 
    else :
      rrx = self.rcvs.x[ self.ircvs ]
      rry = self.rcvs.y[ self.ircvs ]
      ssx = self.srcs.x[ self.isrcs ]
      ssy = self.srcs.y[ self.isrcs ]

    self.offset2d = np.sqrt( ( rry-ssy ) **2 +( rrx-ssx ) **2  ) 
    if sign == 1 :
      self.offset2d *= np.sign( rrx-ssx )

    #}}}}}
  def set_offset( self ) : #{{{{{

    if self.isrcs.ndim > 1 :
      rrx, ssx = np.meshgrid( self.rcvs.x, self.srcs.x ) 
      rry, ssy = np.meshgrid( self.rcvs.y, self.srcs.y ) 
      rrz, ssz = np.meshgrid( self.rcvs.z, self.srcs.z )
    else :
      rrx = self.rcvs.x[ self.ircvs ]
      rry = self.rcvs.y[ self.ircvs ]
      rrz = self.rcvs.z[ self.ircvs ]
      ssx = self.srcs.x[ self.isrcs ]
      ssy = self.srcs.y[ self.isrcs ]
      ssz = self.srcs.z[ self.isrcs ]

    self.offset = np.sqrt( ( rry-ssy ) **2 +( rrx-ssx ) **2  + ( rrz-ssz ) ** 2 ) 
    #}}}}}

  def set_offset3d( self ) : #{{{{{
    self.set_offset() 
#    rrx, ssx = np.meshgrid( self.rcvs.x, self.srcs.x ) 
#    rrz, ssz = np.meshgrid( self.rcvs.z, self.srcs.z )
#    rry, ssy = np.meshgrid( self.rcvs.y, self.srcs.y )
#    self.offset = np.sqrt( ( rrx-ssx ) **2  + ( rrz-ssz ) ** 2 
#                          +( rry-ssy ) **2 ) 
    #}}}}}

  def set_default_fnames( self, fhead, flag_fbreak=1 ) : #{{{{{
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader    = fh+'.header'
    self.fbin       = fh+'.bin'
    self.srcs.fname = fh+'.src'
    self.rcvs.fname = fh+'.rcv'
    if flag_fbreak == 1 :
      self.fbreak.fname = fh+'.fbreak'
    return fh
 #}}}}}

class rn_binary(binary) :
  def __init__( self, ref=None,  nrcv=1, nsrc=1, ot=0, nt=1, dt=1., srcs=None, rcvs=None, flag_init=1 ) :  #{{{{{
    if srcs :
      nsrc = srcs.n
    if rcvs :
      nrcv = rcvs.n
    binary.__init__( self, ref=ref, nrcv=nrcv, nsrc=nsrc ) 

    if ref is None :
      self.ot = ot
      self.dt = dt
      self.nt = nt
    else :
      self.ot = ref.ot
      self.dt = ref.dt
      self.nt = ref.nt 

    self.set_t()
    if flag_init == 1 :
      self.initialize()
  #}}}}}
  
  def initialize( self, val=0., gather=None ) : #{{{{{
    if gather :
      self.gather = gather 
    elif not self.gather   :
      self.gather = 'all'     
  
    if self.gather == 'all' :
      self.d = np.ones( ( self.srcs.n, self.rcvs.n, self.nt ), 
                             dtype=np.float32 ) * val

    elif self.gather == 'shot' :
      self.d = np.ones( ( self.rcvs.n, self.nt ), dtype=np.float32 ) * val

    elif self.gather == 'receiver' :
      self.d = np.ones( ( self.srcs.n, self.nt ), dtype=np.float32 ) * val

    else :
      self.d = np.ones( ( self.srcs.n, self.nt ),
                             dtype=np.float32 ) * val
  #}}}}}

  def read_header(self, fheader=None ): #{{{{{
    binary.read_header( self, fheader )

    #print( self.fdir, self.fheader )

    with open( os.path.join( self.fdir, self.fheader )) as f:
      lines = f.read().splitlines()


    # line 0 bin file information
    self.fbin = lines[0]

    # line 1 time information
    tmp = lines[1].split()
    self.dt = float( tmp[1] )
    self.ot = float( tmp[2] )
    self.nt = int( tmp[0] )
    self.set_t() 


    # line 2 shot information
    self.srcs.fname = lines[2]
    self.srcs.fdir  = self.fdir


    #print self.srcs.fname
    try :
      self.srcs.read()
    except : 
      print( '%s %s:source coord file is not in the right format'%( 
            self.fdir, self.fheader ) )
    # line 3 receiver information
    self.rcvs.fname = lines[3]
    self.rcvs.fdir  = self.fdir  
  
    try :
      self.rcvs.read()
    except : 
      print( '%s %s:receiver coord file is not in the right format'%( 
              self.fdir, self.fheader ) )

    self.ircvs, self.isrcs = np.meshgrid( 
                              np.arange( self.rcvs.n, dtype=int ),
                              np.arange( self.srcs.n, dtype=int ) )

    # line 4 fbreak information ( optional ) ....
    self.fbreak = rn_fbreak()
    try :
      self.fbreak.fname = lines[4]
      self.fbreak.fdir  = self.fdir
      self.fbreak.read( self.srcs, self.rcvs )
    except :
      self.fbreak.fname = None
      self.fbreak.time = np.ma.masked_equal( 
                      np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=float ), 
                      0.0 ) 
    # line 5 CMP file header infromation

    self.cmps = rn_loc()
    self.offsets = rn_loc()
    try :
      self.cmps.fname = lines[5].split()[0]
      self.offsets.fname = lines[5].split()[1]
    except :
      self.cmps.fname = None
      self.offsets.fname = None
      #}}}}}

  def set_t( self ) : #{{{{{
    self.t = np.arange( 0., self.nt, dtype=float) * self.dt + self.ot
  #}}}}}

  def extract_time( self, tmin, tmax ) : #{{{{{
    self.set_t()
    
    itmin = np.argmin( np.abs( self.t - tmin ) )
    itmax = np.argmin( np.abs( self.t - tmax) )
    print( itmin, itmax )

    if self.d.ndim == 2 :
      self.d = self.d[ :, itmin:(itmax+1) ]
    else:
      self.d = self.d[ :, :, itmin:(itmax+1) ]


    self.ot = self.t[itmin]
    self.nt = itmax - itmin + 1
    self.set_t()
    #}}}}}

  def extend_time( self, tmax ) : #{{{{{
    
    nt = int( round( ( tmax - self.ot ) / self.dt ) )

    tmp = self.d
    
    self.nt = nt

    if self.d.ndim == 2 :
      ntrace = tmp.shape[0]
      self.d = np.zeros( ( ntrace, nt ), dtype=float )
      self.d[ :, :self.nt ] = tmp
    else:
      n0 = tmp.shape[0]
      n1 = tmp.shape[1]
      self.d = np.zeros( ( n0, n1,  nt ), dtype=float )
      self.d[ :, :, :self.nt ] = tmp


    self.set_t()
  #}}}}}


  def write_header(self, fheader=None): #{{{{{
    
    if fheader is not None :
      self.fheader = fheader
      self.fdir = os.path.dirname( self.fheader )
      self.fheader = os.path.basename( self.fheader )

    # make sure to havea corect one
    self.srcs.fdir = self.fdir
    self.rcvs.fdir = self.fdir
    self.fbreak.fdir = self.fdir 


    outlines = []
    outlines.append( '%s'%( self.fbin ) )
    outlines.append( '%d %e %e'%( self.nt, self.dt, self.ot ) )
    outlines.append( '%s'%( self.srcs.fname ) )
    outlines.append( '%s'%( self.rcvs.fname ) )
    if self.fbreak.fname is not None : 
      outlines.append( '%s'%( self.fbreak.fname ) )



    with open( os.path.join( self.fdir, self.fheader ), 'w' ) as f :
      f.write( '\n'.join(outlines) )

    self.srcs.write()
    self.rcvs.write()

    if self.fbreak.fname is not None :
      self.fbreak.write( self.srcs, self.rcvs )
  #}}}}}

  def read_data( self, idx=None, z=None, dtype=np.float32  ) : #{{{{{
    if self.gather == 'shot' :
      self.read_data_shot( ishot=idx, dtype=dtype ) 
    elif self.gather == 'receiver' :
      self.read_data_receiver( ircv=idx, dtype=dtype )
    elif self.gather == 'cmp' :
      self.read_data_cmp( icmp=idx, dtype=dtype ) 
    elif self.gather == 'offset' :
      self.read_data_offset( ioffset=idx, dtype=dtype )

    elif self.gather == 'samelevel' :
      self.read_data_samelevel()
    else :
      self.d = np.fromfile( os.path.join( self.fdir, self.fbin ), 
              dtype = dtype ).reshape(self.nc, self.srcs.n, self.rcvs.n, self.nt)
  #}}}}}


  
  def read_data_shot( self, ishot, dtype=np.float32 ) : # ishot start from zero #{{{{{

    if not self.fbinh :
      self.open_data( op='r' )

    self.isrc = ishot

    try :
      self.fbreak.time = self.fbreak.time[ ishot, : ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.time = np.zeros( self.rcvs.n, dtype=float )

    ibyte  = np.dtype( np.float32 ).itemsize

    self.d = np.zeros( ( self.rcvs.n, self.nt ), dtype=dtype  )

    self.fbinh.seek( ibyte * ishot * self.nt * self.rcvs.n, os.SEEK_SET ) 
    self.d[ :, : ] = np.fromfile( self.fbinh, dtype = dtype, 
                             count= self.nt * self.rcvs.n 
                            ).reshape( self.rcvs.n, self.nt )
    #}}}}} 

  def read_data_receiver( self, ircv, dtype=np.float32) : # ircv starts from 0 #{{{{{
    print( 'receiver gather ')
    if not self.fbinh :
      self.open_data( op='r' )
    ibyte  = np.dtype( dtype ).itemsize
    if self.sort == 'receiver' :
      self.fbinh.seek( ibyte * ircv * self.nt * self.srcs.n, os.SEEK_SET ) 
      self.d[ :, : ] = np.fromfile( self.fbinh, dtype = dtype,  
                               count= self.nt * self.srcs.n 
                              ).reshape( self.srcs.n, self.nt )
    else :
      self.fbinh.seek( ibyte * ircv * self.nt, os.SEEK_SET )
      for isrc in range( self.srcs.n ) :
        self.d[ isrc, : ] = np.fromfile( self.fbinh, dtype = dtype,
                               count= self.nt )
        #print isrc, self.fbinh.tell()
        self.fbinh.seek( ibyte * ( self.rcvs.n -1 ) * self.nt, os.SEEK_CUR )

    try :
      self.fbreak.time = self.fbreak.time[ :, ircv ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.time = np.zeros( self.srcs.n, dtype=float )
    #}}}}}

  def read_data_traces( self, traces, dtype=np.float32 ) : # ircv starts from 0 #{{{{{
    if not self.fbinh :
      self.open_data( op='r' )
      
    ntrace = len( traces )
    ibyte  = np.dtype( dtype ).itemsize

    self.d = np.zeros( ( ntrace, self.nt ), dtype=dtype )
    print( self.d.shape )

    for itrace in range( ntrace ) :
      traceno = traces[itrace]
      print( itrace, traceno)
      self.fbinh.seek( ibyte*traceno*self.nt, os.SEEK_SET )
      self.d[ itrace, : ] = np.fromfile( self.fbinh, dtype=dtype,
                                count = self.nt )

    try :
      self.fbreak.time = self.fbreak.time.reshape( self.nsrc*self.nrcv 
                            )[ traces ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.time = np.zeros( ntrace, dtype=float )
    #}}}}}



  ## These are for nagaoka dataset #{{{{{

  def find_ioffset( self, offset ) :
    return np.argmin( np.abs( self.offsets.z - offset ) )
  def find_icmp( self, zcmp ) :
    return np.argmin( np.abs( self.cmps.z - cmps ) )
  def read_data_offset( self, ioffset=0 ) :

    isrcs, ircvs = np.where( self.ioffset == ioffset ) 

 
    ntraces = len( isrcs )
    self.d = np.zeros( (  ntraces, self.nt ), dtype=float )

    self.zcmp  = self.cmps.z[ self.icmp[ isrcs, ircvs ] ]   

    fbreak = self.fbreak.time[ isrcs, ircvs ]
 
    self.fbreak.time = fbreak

    self.open_data()

    for idx in range( ntraces ) :
      itrace = isrcs[ idx ] * self.rcvs.n + ircvs[ idx ] 
      self.fbinh.seek( 4 * itrace * self.nt , os.SEEK_SET )
      self.d[ idx, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )

  def read_data_cmp( self, icmp=0 ) :

    #icmp = np.argmin( np.abs( self.cmps.z - zcmp ) )


    isrcs, ircvs = np.where( self.icmp == icmp ) 
  
    ntraces = len( isrcs )
    self.d = np.zeros( (  ntraces, self.nt ), dtype=float )

    self.zoffset  = self.offsets.z[ self.ioffset[ isrcs, ircvs ] ]   

    fbreak = self.fbreak.time[ isrcs, ircvs ]
 
    self.fbreak.time = fbreak

    self.open_data()

    for idx in range( ntraces ) :
      itrace = isrcs[ idx ] * self.rcvs.n + ircvs[ idx ] 
      self.fbinh.seek( 4 * itrace * self.nt , os.SEEK_SET )
      self.d[ idx, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )

 
  # this only worns for nagaoka



  def read_data_samelevel( self ) :
    ircvs = np.zeros( self.srcs.n, dtype=int )

    for isrc, zsrc in enumerate( self.srcs.z ) : 
      ircvs[ isrc ] = np.argmin( np.abs( self.rcvs.z - zsrc ) )
      fbreak[ isrc ] = self.fbreak.time[ isrc, ircvs[ isrc ] ] 
  
    self.fbreak.time = fbreak

    self.open_data()

    for isrc in range( self.srcs.n ) :
      #print isrc, ircvs[isrc]
      self.fbinh.seek( 4 * ircvs[ isrc ] * self.nt, os.SEEK_CUR )

      self.d[ isrc, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )
      self.fbinh.seek( 4 * ( self.rcvs.n - ircvs[isrc] - 1 ) * self.nt,
                      os.SEEK_CUR )
  def read_data_samelevel_angle( self, angles ) :
    ircvs = np.zeros( self.srcs.n, dtype=int )
    for isrc, zsrc in enumerate( self.srcs.z ) : 
      zrcvs_possible = zsrc + ( np.tan( angles[ isrc ] ) 
                              * ( self.rcvs.x - self.srcs.x[ isrc ] ) )
      #print zsrc, zrcvs_possible 
      ircvs[ isrc ] = np.argmin( np.abs( self.rcvs.z - zrcvs_possible ) )

    for isrc in range( self.srcs.n ) :
      #print isrc, ircvs[isrc]
      self.fbinh.seek( 4 * ircvs[ isrc ] * self.nt, os.SEEK_CUR )

      self.d[ isrc, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )
      self.fbinh.seek( 4 * ( self.rcvs.n - ircvs[isrc] - 1 ) * self.nt,
                      os.SEEK_CUR )

  #}}}}}


  
  def open_data( self, op='r' ): #{{{{{
    self.fbinh = open( os.path.join( self.fdir, self.fbin ), op+'b' )

  def write_data_ch( self, ich ) :
    if not self.fbinh :
      self.open_data( op='w' )
    self.fbinh.seek( 4 * ich * self.nsamples, os.SEEK_SET ) 
    self.d.astype( np.float32).tofile( self.fbinh )

  def close_data( self ) :
    self.fbinh.close()


  def write_data_append( self ) : #{{{{{
    if not self.fbinh :
      self.open_data() 
    self.d.astype( np.float32 ).tofile( self.fbinh )

  #}}}}}


  def write_data( self ) : #{{{{{

    self.d.astype(np.float32).tofile( os.path.join( self.fdir, self.fbin ))   

  # }}}}}

  def write( self ) : #{{{{{
    self.write_header()
    self.write_data()


  #}}}}}

  def set_rms( self ) : #{{{{{
    self.rms = np.sum( self.d**2, axis=-1 ) / self.nt
  #}}}}}


class cbinary( binary ) :
  def __init__( self, ref=None, nrcv=1, nsrc=1, nfreq=1, srcs=None, rcvs=None, freqs=None ) : #{{{{{
    binary.__init__( self, ref, nrcv, nsrc, srcs, rcvs )

    if ref is None :
      if freqs is None :
        self.freqs = rn_freq( n=nfreq )
      else :
        self.freqs = freqs 
    else :
      self.freqs = copy.deepcopy( ref.freqs )

    self.nf = self.freqs.n
   
    self.d = None
    self.initialize()
     # }}}}}

  def initialize( self, val=0., gather=None ) : #{{{{{
    if gather :
      self.gather = gather 
    elif not self.gather   :
      self.gather = 'all'     
  
    if self.gather == 'all' :
      if type( self.d ) is np.ndarray :
        self.d[:,:,:] = val
      else :
        self.d = np.ones( ( self.srcs.n, self.rcvs.n, self.freqs.n ), 
                               dtype=np.complex64 ) * val

    elif self.gather == 'shot' :
      self.d = np.ones( ( self.rcvs.n, self.nf ), dtype=np.complex64 ) * val

    elif self.gather == 'receiver' :
      self.d = np.ones( ( self.srcs.n, self.nf ), dtype=np.complex64 ) * val

    else :
      self.d = np.ones( ( self.srcs.n, self.nf ),
                             dtype=np.complex64 ) * val
  #}}}}}

  def set_default_fnames( self, fhead, flag_fbreak=1 ) : #{{{{{
    fh = binary.set_default_fnames( self, fhead, flag_fbreak=flag_fbreak )
    self.freqs.fname = fh + '.freq'
  def read_header(self, fheader=None ): #{{{{{
    binary.read_header( self, fheader )

    with open( os.path.join( self.fdir, self.fheader )) as f:
      lines = f.read().splitlines()

    # line 0 bin file information
    self.fbin = lines[0]

    # line 1 frequency information
    self.freqs.fname = lines[1]
    self.freqs.fdir = self.fdir 

    try :
      self.freqs.read()
    except :
      print ('%s, %s: frequency file is not avalable/not in the right format'%
              ( self.fdir, self.fheader  ) )

    # line 2 shot information
    self.srcs.fname = lines[2]
    self.srcs.fdir  = self.fdir


    #print self.srcs.fname
    try :
      self.srcs.read()
    except : 
      print( '%s %s:source coord file is not in the right format'%(  
            self.fdir, self.fheader ))
    # line 3 receiver information
    self.rcvs.fname = lines[3]
    self.rcvs.fdir  = self.fdir  
  
    try :
      self.rcvs.read()
    except : 
      print( '%s %s:receiver coord file is not in the right format'%( 
              self.fdir, self.fheader ) )

    self.ircvs, self.isrcs = np.meshgrid( 
                              np.arange( self.rcvs.n, dtype=int ),
                              np.arange( self.srcs.n, dtype=int ) )

    # line 4 fbreak information ( optional ) ....
    self.fbreak = rn_fbreak()
    try :
      self.fbreak.fname = lines[4]
      self.fbreak.fdir  = self.fdir
      self.fbreak.read( self.srcs, self.rcvs )
    except :
      self.fbreak.fname = None
      self.fbreak.time = np.ma.masked_equal( 
                      np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=float ), 
                      0.0 ) 
    # line 5 CMP file header infromation

    self.cmps = rn_loc()
    self.offsets = rn_loc()
    try :
      self.cmps.fname = lines[5].split()[0]
      self.offsets.fname = lines[5].split()[1]
    except :
      self.cmps.fname = None
      self.offsets.fname = None
      #}}}}}

  def read_data( self, idx=None, z=None  ) : #{{{{{
    if self.gather == 'shot' :
      self.read_data_shot( ishot=idx ) 
    elif self.gather == 'receiver' :
      self.read_data_receiver( ircv=idx )
    elif self.gather == 'cmp' :
      self.read_data_cmp( icmp=idx ) 
    elif self.gather == 'offset' :
      self.read_data_offset( ioffset=idx )

    elif self.gather == 'samelevel' :
      self.read_data_samelevel()
    else :
      self.d = np.fromfile( os.path.join( self.fdir, self.fbin ), 
              dtype = np.dtype('complex64') 
             ).reshape(self.srcs.n, self.rcvs.n, self.freqs.n)

    #}}}}}

  def write_header(self, fheader=None): #{{{{{

    
    if fheader is not None :
      self.fheader = fheader
      self.fdir = os.path.dirname( self.fheader )
      self.fheader = os.path.basename( self.fheader )

    # make sure to havea corect one
    self.srcs.fdir = self.fdir
    self.rcvs.fdir = self.fdir
    self.fbreak.fdir = self.fdir 
    self.freqs.fdir = self.fdir 


    outlines = []
    outlines.append( '%s'%( self.fbin ) )
    outlines.append( '%s'%( self.freqs.fname ) )
    outlines.append( '%s'%( self.srcs.fname ) )
    outlines.append( '%s'%( self.rcvs.fname ) )
    if self.fbreak.fname is not None : 
      outlines.append( '%s'%( self.fbreak.fname ) )


    with open( os.path.join( self.fdir, self.fheader ), 'w' ) as f :
      f.write( '\n'.join(outlines) )

    self.srcs.write()
    self.rcvs.write()
    self.freqs.write()

    self.fbreak.write( self.srcs, self.rcvs )
  #}}}}}

  def write_data( self ) : #{{{{{
    self.d.astype('complex64').tofile( os.path.join( self.fdir, self.fbin ))   
  #}}}}}

  def write( self ) : #{{{{{
    self.write_header()
    self.write_data()
    #}}}}}

