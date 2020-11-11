#!/usr/bin/env python
import datetime
import numpy as np
import pytz 
from dateutil import parser
import os


import rn.libs.normalize as rn_normalize


tz_jp = pytz.timezone('Asia/Tokyo')
#===========================================================================
# Location Object
#===========================================================================


class rn_loc( object ) :
  def __init__(self, n=1 ) : 
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
  
  def set_n( self, n=1 ) :
    self.n  = n
 
  def initialize( self, val=0.) :
    self.x = np.zeros( self.n, dtype=np.float )
    self.y = np.zeros( self.n, dtype=np.float )
    self.z = np.zeros( self.n, dtype=np.float )
    self.id = []
    self.time = []
    for i in range( self.n ) :
      self.id.append( 'id-%d'%i )
      self.time.append( datetime.datetime( 1900, 1, 1, tzinfo = tz_jp ) )

  def read( self ) :
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


    try :
      self.time = [  tz_jp.localize( datetime.datetime.strptime( 
                      line.split()[4], '%Y-%m-%d-%H:%M:%S:%f'
                       ) )
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

#===========================================================================
# Binary object
#===========================================================================
class rn_binary(object) :
  def __init__(self):
    self.samp_freq = 250
    self.nchannels = 4
    self.fdir =  ''


#    self.channels  = []
    self.t0    = datetime.datetime.today()
    self.t1    = datetime.datetime.today()
    self.data    = np.zeros( ( self.nchannels, self.samp_freq) )
    self.totalsec  = 0
    self.nsamples  = 0
    self.data    = None


  
    self.chs = rn_loc()

  def initialize(self, val = 0. ) :
    self.totalsec = int( (self.t1 - self.t0 ).total_seconds() ) + 1 
    self.nsamples = int( self.samp_freq * self.totalsec )
    self.data  = np.ones( ( self.nchannels, self.nsamples ) ,
                dtype = 'float_') * val


  # part of convert from Win format 
  def read_from_info(self, finfo):
    with open(finfo) as f:
      info = f.read().splitlines()
   
    # get sampling frequency
    self.samp_freq = int(info[0].split()[4])

    n = 0
    for i in range(len(info)) :
      if not info[i] :
        n = i
        break

    print 'empty line found at ', n

    # now find # of components
    self.nchannels = int(info[n+1].split()[4])
    print 'number of channels: ', self.nchannels
    
    self.channels = [ x.split()[3] for x in info[ 0:self.nchannels ] ] 

    print 'channels: ', self.channels

    # now find start time 
    date0 = info[ self.nchannels+1 ].split()[2]
    time0 = info[ self.nchannels+1 ].split()[3]
    print 'start time: ', date0, time0
    try :
      self.t0 = parser.parse(date0+time0+'+0900')
    except :
      self.t0 = parser.parse('20'+date0+time0+'+0900')

    # now find end time
    date1 = info[ -4 ].split()[2]
    time1 = info[ -4 ].split()[3]
    self.t1 = parser.parse('20'+date1+time1+'+0900')
    
    print 'end time: ', date1, time1

  # this is left for some historical reason
  def read_header_nofbin(self, fheader=None, sett=1 ):

    if fheader is not None :
      self.fheader = fheader

    self.fdir = os.path.dirname( self.fheader )
    self.fheader = os.path.basename( self.fheader )

    with open( os.path.join( self.fdir, self.fheader) ) as f:
      self.samp_freq = int( float( f.readline() ) )
      (year0, mon0, mday0, yday0, hour0, min0, sec0) = [  int(x) 
                        for x in f.readline().split() ]
      self.t0 = tz_jp.localize( 
                  datetime.datetime(year0, mon0, mday0, hour0, min0, sec0 ) )
      self.yday0 = yday0
      (year1, mon1, mday1, yday1, hour1, min1, sec1) = [  int(x) 
                        for x in f.readline().split() ]
      self.t1 = tz_jp.localize( 
                  datetime.datetime(year1, mon1, mday1, hour1, min1, sec1 ) )
      self.yday1 = yday1
      self.nchannels = int(f.readline())

      self.channels  = f.readline().split()

      self.chs = rn_loc( n=self.nchannels )
      self.chs.id = self.channels
      self.chs.x  = np.zeros( self.chs.n, dtype=np.float )
      self.chs.y  = np.zeros( self.chs.n, dtype=np.float )
      self.chs.z  = np.zeros( self.chs.n, dtype=np.float )

      self.set_totalsec_nsamples()


  def read_header(self, fheader=None ):
    if fheader is not None :
      self.fheader = fheader

    self.fdir = os.path.dirname( self.fheader )
    self.chs.fdir = self.fdir

    self.fheader = os.path.basename( self.fheader )
    with open( os.path.join( self.fdir, self.fheader) ) as f:
      self.fbin = f.readline().rstrip()
      self.samp_freq = int( float( f.readline() ) )
      (year0, mon0, mday0, yday0, hour0, min0, sec0) = [  int(x) 
                        for x in f.readline().split() ]
      self.t0 = tz_jp.localize(
              datetime.datetime(year0, mon0, mday0, hour0, min0, sec0 ) )
      self.yday0 = yday0
      (year1, mon1, mday1, yday1, hour1, min1, sec1) = [  int(x) 
                        for x in f.readline().split() ]
      self.t1 = tz_jp.localize( 
              datetime.datetime(year1, mon1, mday1, hour1, min1, sec1 ))
      self.yday1 = yday1
      ( self.nchannels, self.chs.fname ) = f.readline().rstrip().split()
      self.chs.fname = self.chs.fname.rstrip()
      self.nchannels = int( self.nchannels )

      self.chs.read( )

      self.set_totalsec_nsamples()

    self.set_t()


  def set_totalsec_nsamples( self ) :
    self.totalsec = int( (self.t1 - self.t0 ).total_seconds() ) + 1 
    self.nsamples = int( self.samp_freq * self.totalsec )

  def set_t( self ) :
    self.t = []
    for it in range( self.nsamples ) :
      self.t.append( self.t0 + datetime.timedelta( 0, float(it) / self.samp_freq ) )



  def set_default_fnames( self, fhead ) :
    self.fdir = os.path.dirname( fhead )
    fh = os.path.basename( fhead )

    self.fheader    = fh+'.header'
    self.fbin       = fh+'.bin'
    self.chs.fdir = self.fdir
    self.chs.fname = fh+'.ch'

  def write_header(self, fheader=None ):

    if fheader is not None :
      self.fheader = fheader
      self.fdir = os.path.dirname( self.fheader )
      self.fheader = os.path.basename( self.fheader )


    self.set_totalsec_nsamples()
    t0 = self.t0.timetuple()
    t1 = self.t1.timetuple()

    outline=[]
    outline.append('%s'%self.fbin)
    outline.append('%f'%self.samp_freq)
    outline.append('%4d %02d %02d %03d %02d %02d %02d'%( 
            t0.tm_year, t0.tm_mon, t0.tm_mday, t0.tm_yday, 
            t0.tm_hour, t0.tm_min, t0.tm_sec))
    outline.append('%4d %02d %02d %03d %02d %02d %02d'%( 
            t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_yday, 
            t1.tm_hour, t1.tm_min, t1.tm_sec))
    outline.append('%d %s'%(self.nchannels, self.chs.fname ))
    #outline.append(' '.join(self.channels))
    outline.append('%d %f'%( self.nsamples, self.totalsec ))

    self.chs.write()


    with open( os.path.join( self.fdir, self.fheader ),'w') as f:
      f.write('\n'.join(outline))

  def print_header_tab(self):
    self.set_totalsec_nsamples()
    t0 = self.t0.timetuple()
    t1 = self.t1.timetuple()

    outline=[]
    outline.append('%f'%self.samp_freq)
    outline.append('%4d\t%03d\t%02d\t%02d\t%02d'%( 
            t0.tm_year, t0.tm_yday, 
            t0.tm_hour, t0.tm_min, t0.tm_sec))
    outline.append('%4d\t%03d\t%02d\t%02d\t%02d'%( 
            t1.tm_year, t1.tm_yday, 
            t1.tm_hour, t1.tm_min, t1.tm_sec))
    outline.append('%d'%self.nchannels)
    if self.nchannels < 4 : 
      self.channels.append(' ')
    outline.append('\t'.join(self.channels))
    outline.append('%d\t%f'%( self.nsamples, self.totalsec ))
    print('\t'.join(outline))


  # use this if read after win2bin
  def read_data_int(self):
    self.set_totalsec_nsamples() 
    print self.nsamples,self.nchannels
    self.data = np.fromfile( os.path.join( self.fdir, self.fbin ), 
          dtype = np.dtype('i4') 
           ).reshape(self.chs.n, self.nsamples)

  # use this if data is binary
  def read_data( self,fbin=None) :
    if fbin is not None :
      self.fbin = fbin
    self.set_totalsec_nsamples() 
    print self.nsamples,self.nchannels
    self.data = np.fromfile( os.path.join( self.fdir, self.fbin ), 
            dtype = np.dtype('float32') 
           ).reshape(self.ch.n, self.nsamples)

  def extract_time( self, t0, t1 ) :
    self.set_t()
    
    if t0 > self.t0 :
      it0 = self.t.index( t0 ) 
      self.t0 = self.t[ it0 ];
    else :
      it0 = 0
    
    if t1 < self.t1 :
      it1 = self.t.index( t1 ) 
#      it1 = int( ( ( t1 - self.t0 ).total_seconds() +1 )* self.samp_freq )
      self.t1 = self.t[it1]
      it1 = it1 + 1
    else :
      it1 = self.nsamples - it0


    self.set_totalsec_nsamples()
    self.data = self.data.transpose()[ it0 : it1, : ].transpose()



  def extract_channel( self, ch ) :
    ich = self.channels.index( ch )
    self.nchannels = 1
    self.channels = [ ch ]
    self.data = self.data[ ich, : ].reshape( ( 1, self.nsamples) )


  # write ata
  def write_data( self, fbin=None ) :
    if fbin is not None :
      self.fbin = fbin

    self.data.astype('float32').tofile( os.path.join( self.fdir, self.fbin ) )   

