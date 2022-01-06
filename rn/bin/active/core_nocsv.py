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
  print( 'utm was not found' )


tz_jp = pytz.timezone('Asia/Tokyo')


class rn_loc( object ) : #{{{{{
  def __init__(self, n=1, ref=None ) : 
    if ref : 
      self.n = ref.n
      self.x = ref.x
      self.y = ref.y
      self.z = ref.z
      try :
        self.fname = self.fname 
      except :
        self.fname = ''
      try :
        self.fdir = self.fdir
      except :
        self.fdir = ''
      try :
        self.id = self.id
      except :
        self.id = []
        for i in range( self.n )  :
          self.id.append( '%d'%i )
      try :
        self.md = self.md
      except :
        self.md = []
    else :
      self.n  = n
      self.x  = np.zeros(self.n)
      self.y  = np.zeros(self.n)
      self.z  = np.zeros(self.n) 
      self.fname = ''
      self.fdir  = ''
      self.id = []
      self.md = []
      for i in range( self.n )  :
        self.id.append( '%d'%i )


  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fname = os.path.basename( fname )

  def set_n( self, n=1 ) :
    self.n  = n

  def set_regular( self, n=None, ox=0.,dx=1.,oz=0.,dz=1.) :
    if n :
      self.set_n( n=n )
    if dx == 0 :
      self.x = np.ones( self.n, dtype=np.float ) * ox 
    else :
      self.x = np.arange( self.n, dtype=np.float ) * dx + ox
    if dz == 0 :
      self.z = np.ones( self.n, dtype=np.float ) * oz 
    else :
      self.z = np.arange( self.n, dtype=np.float ) * dz + oz

  def initialize( self, val=0. ) :
    self.x = np.ones( self.n, dtype=np.float ) * val
    self.y = np.ones( self.n, dtype=np.float ) * val
    self.z = np.ones( self.n, dtype=np.float ) * val
    self.id = []
    self.time = []
    for i in range( self.n ) :
      self.id.append( 'id-%d'%i )
      self.time.append( datetime.datetime( 1900, 1, 1, tzinfo = tz_jp ) )

  def read( self, fname=None) :
    if fname :
      self.set_fname( fname )

    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.set_n( len(lines) )

    self.id = np.array( [ line.split()[0] for line in lines ],
                        dtype=np.unicode_ )
    self.x  = np.array( [ line.split()[1] for line in lines ], 
                        dtype=np.float ) 
    self.y  = np.array( [ line.split()[2] for line in lines ], 
                        dtype=np.float ) 
    self.z  = np.array( [line.split()[3]  for line in lines ],
                        dtype=np.float )

    # nagaoka 
    try : 
      self.md = self.id.astype( np.float )
    except :
      self.md = self.z


    try :
      self.time = [ datetime.datetime.strptime( 
                      line.split()[4], '%Y-%m-%d-%H:%M:%S:%f',
                      tzinfo = tz_jp )
                      for line in lines ]
    except :
      self.time = None #[ datetime.datetime( 1900, 1, 1, tzinfo = tz_jp) for line in lines ]

  def write( self ) :
    outlines = []
    try :
      for i in range(self.n) :
        outlines.append( '%s %f %f %f %s' % 
                           ( self.id[i], 
                             self.x[i], self.y[i], self.z[i],
                             self.time[i].strftime('%Y-%m-%d-%H:%M:%S:%f')
                            )
                       )
    except :
      for i in range(self.n) :
        outlines.append( '%s %f %f %f' % 
                             ( self.id[i], 
                               self.x[i], self.y[i], self.z[i]
                              )
                         )
   

    with open( os.path.join( self.fdir, self.fname ), 'w' ) as f :
      f.write( '\n'.join( outlines ) )

  def latlon2utm( self, lat='x' ) :
    try :
      for i in range( self.n ) :
        tmp = utm.from_latlon( self.y[i], self.x[i] )
        self.x[i] = tmp[0]
        self.y[i] = tmp[1]
    except :
      for i in range( self.n ) :
        tmp = utm.from_latlon( self.x[i], self.y[i] )
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
    self.offset = np.zeros( self.n, dtype=np.float )
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
                        dtype=np.float ) 


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
    self.times    = []
    self.fname = None #'test.fbreak'  

  def initialize( self, srcs, rcvs ) :
    self.times = np.ma.zeros( ( srcs.n, rcvs.n ), dtype=np.float )
    self.srcsid = np.zeros( ( srcs.n, rcvs.n ), dtype=np.unicode_)
    self.rcvsid = np.zeros( ( srcs.n, rcvs.n ), dtype=np.unicode_)
    

  def set_fname( self, fname ) :
    self.fdir = os.path.dirname( fname )
    self.fname = os.path.basename( fname )
  def read( self, srcs, rcvs ) :
    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.srcsid = np.array( [ line.split()[0] for line in lines ], 
                       dtype=np.unicode_ ).reshape( srcs.n, rcvs.n )
    self.rcvsid = np.array( [ line.split()[1] for line in lines ],
                       dtype=np.unicode_ ).reshape( srcs.n, rcvs.n )
    self.times  = np.array( [ line.split()[2] for line in lines ], 
                       dtype=np.float ).reshape( srcs.n, rcvs.n )
    self.times  = np.ma.masked_equal( self.times, 0.0 ) 

    #print self.srcsid, self.rcvsid, self.time
              
  
  def write( self, srcs, rcvs ) :

    if self.fname :
      outlines = []
      times_filled = self.times.filled( 0.0 )
      for isrc in range( srcs.n ) :
        for ircv in range( rcvs.n ) :
          outlines.append( '%s %s %f'%(srcs.id[ isrc ],
                                       rcvs.id[ ircv ],
                                       times_filled[ isrc, ircv ] ))
      with open( os.path.join( self.fdir, self.fname ), 'w' ) as f :
        f.write( '\n'.join( outlines ) )


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
    self.d = np.ones( self.n, dtype=np.float ) * val

  def read( self ) :
    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.set_n( len(lines) )

    self.d = np.array( [ line.split()[0] for line in lines ],
                        dtype=np.float )

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
      self.fbreak.initialize( srcs=self.srcs, rcvs=self.rcvs )

      self.isrc = None
      self.ircv = None
      self.gather = 'all'
      self.sort   = 'shot'

      self.fdir    = ''
      self.fbin    = 'test.bin'
      self.fheader = 'test.header' 


      self.ircvs, self.isrcs = np.meshgrid( np.arange( nrcv, dtype=np.int ),
                                  np.arange( nsrc, dtype=np.int ) )

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
  def __init__( self, ref=None,  nrcv=1, nsrc=1, ot=0, nt=1, dt=1., srcs=None, rcvs=None) :  #{{{{{
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
    self.initialize()
  #}}}}}
  
  def initialize( self, val=0., gather=None ) : #{{{{{
    if gather :
      self.gather = gather 
    elif not self.gather   :
      self.gather = 'all'     
  
    if self.gather == 'all' :
      self.data = np.ones( ( self.srcs.n, self.rcvs.n, self.nt ), 
                             dtype=np.float32 ) * val

    elif self.gather == 'shot' :
      self.data = np.ones( ( self.rcvs.n, self.nt ), dtype=np.float32 ) * val

    elif self.gather == 'receiver' :
      self.data = np.ones( ( self.srcs.n, self.nt ), dtype=np.float32 ) * val

    else :
      self.data = np.ones( ( self.srcs.n, self.nt ),
                             dtype=np.float32 ) * val
  #}}}}}

  def read_header(self, fheader=None ): #{{{{{
    binary.read_header( self, fheader )


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
                              np.arange( self.rcvs.n, dtype=np.int ),
                              np.arange( self.srcs.n, dtype=np.int ) )

    # line 4 fbreak information ( optional ) ....
    self.fbreak = rn_fbreak()
    try :
      self.fbreak.fname = lines[4]
      self.fbreak.fdir  = self.fdir
      self.fbreak.read( self.srcs, self.rcvs )
    except :
      self.fbreak.fname = None
      self.fbreak.times = np.ma.masked_equal( 
                      np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=np.float ), 
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
    self.t = np.arange( 0., self.nt, dtype=np.float) * self.dt + self.ot
  #}}}}}

  # nagaoka specific
  def set_cmp_nagaoka( self, ocmp, dcmp, ncmp, ooffset, doffset, noffset,  #{{{{{
                       flagmd=0 ) :
    self.cmps   = rn_loc( ncmp )
    self.offsets = rn_loc( noffset )

    self.cmps.z     = np.arange( ncmp, dtype=np.float ) * dcmp + ocmp
    self.offsets.z  = np.arange( noffset, dtype=np.float ) * doffset + ooffset
    
    if flagmd == 1 :
      rrz, ssz = np.meshgrid( self.rcvs.md, self.srcs.md )
    else :
      rrz, ssz = np.meshgrid( self.rcvs.z, self.srcs.z )
    zcmp = ( rrz + ssz ) / 2. 
    zoffset = rrz - ssz

    self.icmp    = np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=np.int )
    self.ioffset = np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=np.int )
    for isrc in range( self.srcs.n ) :
      for ircv in range( self.rcvs.n ) : 
        self.icmp[ isrc, ircv ] =  np.argmin( np.abs( zcmp[ isrc, ircv ] 
                                                    - self.cmps.z ) )
        self.ioffset[ isrc, ircv ] =  np.argmin( np.abs( zoffset[ isrc, ircv ] 
                                                    - self.offsets.z ) )

  #}}}}}  

  def extract_time( self, tmin, tmax ) : #{{{{{
    self.set_t()
    
    itmin = np.argmin( np.abs( self.t - tmin ) )
    itmax = np.argmin( np.abs( self.t - tmax) )


    if self.data.ndim == 2 :
      self.data = self.data[ :, itmin:(itmax+1) ]
    else:
      self.data = self.data[ :, :, itmin:(itmax+1) ]


    self.ot = self.t[itmin]
    self.nt = itmax - itmin + 1
    self.set_t()
    #}}}}}


  def extend_time( self, tmax ) : #{{{{{
    
    nt = int( round( ( tmax - self.ot ) / self.dt ) )

    tmp = self.data
    

    if self.data.ndim == 2 :
      ntrace = tmp.shape[0]
      self.data = np.zeros( ( ntrace, nt ), dtype=np.float )
      self.data[ :, :self.nt ] = tmp
    else:
      n0 = tmp.shape[0]
      n1 = tmp.shape[1]
      self.data = np.zeros( ( n0, n1,  nt ), dtype=np.float )
      self.data[ :, :, :self.nt ] = tmp


    self.nt = nt
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
      self.data = np.fromfile( os.path.join( self.fdir, self.fbin ), 
              dtype = np.dtype('float32') 
             ).reshape(self.srcs.n, self.rcvs.n, self.nt)
  #}}}}}


  
  def read_data_shot( self, ishot ) : # ishot start from zero #{{{{{

    if not self.fbinh :
      self.open_data( op='r' )

    self.isrc = ishot

    try :
      self.fbreak.times = self.fbreak.times[ ishot, : ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.times = np.zeros( self.rcvs.n, dtype=np.float )


    self.fbinh.seek( 4 * ishot * self.nt * self.rcvs.n, os.SEEK_SET ) 
    self.data[ :, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt * self.rcvs.n 
                            ).reshape( self.rcvs.n, self.nt )
    #}}}}} 

  def read_data_receiver( self, ircv ) : # ircv starts from 0 #{{{{{
    if not self.fbinh :
      self.open_data( op='r' )

    if self.sort == 'receiver' :
      self.fbinh.seek( 4 * ircv * self.nt * self.srcs.n, os.SEEK_SET ) 
      self.data[ :, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                               count= self.nt * self.srcs.n 
                              ).reshape( self.srcs.n, self.nt )
    else :
      self.fbinh.seek( 4 * ircv * self.nt, os.SEEK_SET )
      for isrc in range( self.srcs.n ) :
        self.data[ isrc, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                               count= self.nt )
        #print isrc, self.fbinh.tell()
        self.fbinh.seek( 4 * ( self.rcvs.n -1 ) * self.nt, os.SEEK_CUR )

    try :
      self.fbreak.times = self.fbreak.times[ :, ircv ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.times = np.zeros( self.srcs.n, dtype=np.float )
    #}}}}}

  def read_data_traces( self, traces ) : # ircv starts from 0 #{{{{{
    if not self.fbinh :
      self.open_data( op='r' )
      
    ntrace = len( traces )

    self.data = np.zeros( ( ntrace, self.nt ), dtype=np.float32 )
    print( self.data.shape )

    for itrace in range( ntrace ) :
      traceno = traces[itrace]
      print( itrace, traceno)
      self.fbinh.seek( 4*traceno*self.nt, os.SEEK_SET )
      self.data[ itrace, : ] = np.fromfile( self.fbinh, dtype=np.float32,
                                count = self.nt )

    try :
      self.fbreak.times = self.fbreak.times.reshape( self.nsrc*self.nrcv 
                            )[ traces ]
    except :
      print( 'no fbreak data available' )
      self.fbreak.times = np.zeros( ntrace, dtype=np.float )
    #}}}}}



  ## These are for nagaoka dataset #{{{{{

  def find_ioffset( self, offset ) :
    return np.argmin( np.abs( self.offsets.z - offset ) )
  def find_icmp( self, zcmp ) :
    return np.argmin( np.abs( self.cmps.z - cmps ) )
  def read_data_offset( self, ioffset=0 ) :

    isrcs, ircvs = np.where( self.ioffset == ioffset ) 

 
    ntraces = len( isrcs )
    self.data = np.zeros( (  ntraces, self.nt ), dtype=np.float )

    self.zcmp  = self.cmps.z[ self.icmp[ isrcs, ircvs ] ]   

    fbreak = self.fbreak.times[ isrcs, ircvs ]
 
    self.fbreak.times = fbreak

    self.open_data()

    for idx in range( ntraces ) :
      itrace = isrcs[ idx ] * self.rcvs.n + ircvs[ idx ] 
      self.fbinh.seek( 4 * itrace * self.nt , os.SEEK_SET )
      self.data[ idx, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )

  def read_data_cmp( self, icmp=0 ) :

    #icmp = np.argmin( np.abs( self.cmps.z - zcmp ) )


    isrcs, ircvs = np.where( self.icmp == icmp ) 
  
    ntraces = len( isrcs )
    self.data = np.zeros( (  ntraces, self.nt ), dtype=np.float )

    self.zoffset  = self.offsets.z[ self.ioffset[ isrcs, ircvs ] ]   

    fbreak = self.fbreak.times[ isrcs, ircvs ]
 
    self.fbreak.times = fbreak

    self.open_data()

    for idx in range( ntraces ) :
      itrace = isrcs[ idx ] * self.rcvs.n + ircvs[ idx ] 
      self.fbinh.seek( 4 * itrace * self.nt , os.SEEK_SET )
      self.data[ idx, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )

 
  # this only worns for nagaoka



  def read_data_samelevel( self ) :
    ircvs = np.zeros( self.srcs.n, dtype=np.int )

    for isrc, zsrc in enumerate( self.srcs.z ) : 
      ircvs[ isrc ] = np.argmin( np.abs( self.rcvs.z - zsrc ) )
      fbreak[ isrc ] = self.fbreak.times[ isrc, ircvs[ isrc ] ] 
  
    self.fbreak.times = fbreak

    self.open_data()

    for isrc in range( self.srcs.n ) :
      #print isrc, ircvs[isrc]
      self.fbinh.seek( 4 * ircvs[ isrc ] * self.nt, os.SEEK_CUR )

      self.data[ isrc, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
                             count= self.nt )
      self.fbinh.seek( 4 * ( self.rcvs.n - ircvs[isrc] - 1 ) * self.nt,
                      os.SEEK_CUR )
  def read_data_samelevel_angle( self, angles ) :
    ircvs = np.zeros( self.srcs.n, dtype=np.int )
    for isrc, zsrc in enumerate( self.srcs.z ) : 
      zrcvs_possible = zsrc + ( np.tan( angles[ isrc ] ) 
                              * ( self.rcvs.x - self.srcs.x[ isrc ] ) )
      #print zsrc, zrcvs_possible 
      ircvs[ isrc ] = np.argmin( np.abs( self.rcvs.z - zrcvs_possible ) )

    for isrc in range( self.srcs.n ) :
      #print isrc, ircvs[isrc]
      self.fbinh.seek( 4 * ircvs[ isrc ] * self.nt, os.SEEK_CUR )

      self.data[ isrc, : ] = np.fromfile( self.fbinh, dtype = np.float32, 
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
    self.data.astype( np.float32).tofile( self.fbinh )

  def close_data( self ) :
    self.fbinh.close()


  def write_data_append( self ) : #{{{{{
    if not self.fbinh :
      self.open_data() 
    self.data.astype( np.float32 ).tofile( self.fbinh )

  #}}}}}


  def write_data( self ) : #{{{{{

    self.data.astype('float32').tofile( os.path.join( self.fdir, self.fbin ))   

  # }}}}}

  def write( self ) : #{{{{{
    self.write_header()
    self.write_data()


  #}}}}}

  def set_rms( self ) : #{{{{{
    self.rms = np.sum( self.data**2, axis=-1 ) / self.nt
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
   
    self.data = None
    self.initialize()
     # }}}}}

  def initialize( self, val=0., gather=None ) : #{{{{{
    if gather :
      self.gather = gather 
    elif not self.gather   :
      self.gather = 'all'     
  
    if self.gather == 'all' :
      if type( self.data ) is np.ndarray :
        self.data[:,:,:] = val
      else :
        self.data = np.ones( ( self.srcs.n, self.rcvs.n, self.freqs.n ), 
                               dtype=np.complex64 ) * val

    elif self.gather == 'shot' :
      self.data = np.ones( ( self.rcvs.n, self.nf ), dtype=np.complex64 ) * val

    elif self.gather == 'receiver' :
      self.data = np.ones( ( self.srcs.n, self.nf ), dtype=np.complex64 ) * val

    else :
      self.data = np.ones( ( self.srcs.n, self.nf ),
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
                              np.arange( self.rcvs.n, dtype=np.int ),
                              np.arange( self.srcs.n, dtype=np.int ) )

    # line 4 fbreak information ( optional ) ....
    self.fbreak = rn_fbreak()
    try :
      self.fbreak.fname = lines[4]
      self.fbreak.fdir  = self.fdir
      self.fbreak.read( self.srcs, self.rcvs )
    except :
      self.fbreak.fname = None
      self.fbreak.times = np.ma.masked_equal( 
                      np.zeros( ( self.srcs.n, self.rcvs.n ), dtype=np.float ), 
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
      self.data = np.fromfile( os.path.join( self.fdir, self.fbin ), 
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
    self.data.astype('complex64').tofile( os.path.join( self.fdir, self.fbin ))   
  #}}}}}

  def write( self ) : #{{{{{
    self.write_header()
    self.write_data()
    #}}}}}
