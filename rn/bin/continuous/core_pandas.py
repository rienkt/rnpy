#!/usr/bin/env python
import datetime
import numpy as np
import pytz 
from dateutil import parser
import os
from collections import OrderedDict


import rn.libs.normalize as rn_normalize


tz_jp = pytz.timezone('Asia/Tokyo')
#===========================================================================
# Location Object
#===========================================================================

# station location
class rn_station( object ) :
  def __init__( self, station=None, x=0., y=0., z=0.,
                xname='lon', yname='lat', zname='elev', comps=None, chs=None ) :

    self.station = station
    self.x      = x
    self.y      = y
    self.z      = z
    self.xname  = xname
    self.yname  = yname
    self.zname  = zname
    if comps is None :
      self.comps = []
    else :
      self.comps = comps

    if chs is None :
      self.chs = []
    else :
      self.chs = chs

class rn_stations( object ) :
  def __init__( self, n=1, stations = None ) :
    self.n = n
    self.fdir = ''
    self.fname = ''
    if stations is None :
      self.d = OrderedDict()
    else :
      self.d = OrderedDict()
      for station in stations : 
        self.d[ station ] = rn_station( station )

  def set( self, station=None, x=0., y=0., z=0., xname='lon', yname='lat',
    zname='elev', comps=None, chs=None ):
    if type( station ) is str :
      self.d[ station ] = rn_station( station=station, x=x, y=y, z=z,
                           xname=xname, yname=yname, zname=zname, chs=chs,
                           comps=comps )
    else : 
      self.d[ station.station ] = station

  def remove( self, stations ) :
    if type(stations) is str :
      stations = [ stations ]

    nstation = len( stations )
    for station in stations :
      del self.d[ station ]

    self.n -= nstation

  def write( self ) :
    outlines = []
    stations = self.d.keys()

    for station in stations :
      stainfo = self.d[ station ]
      comps = '-'.join( stainfo.comps )
      chs = '-'.join( stainfo.chs )
      outlines.append( '%s %f %f %f %s %s %s %s %s' % 
                       ( stainfo.station,stainfo.x, stainfo.y, stainfo.z, 
                       stainfo.xname, stainfo.yname, stainfo.zname, 
                      chs,comps ) )

    with open( os.path.join( self.fdir, self.fname ), 'w' ) as f :
      f.write( '\n'.join( outlines ) )



# channel number 
class rn_ch( object ) : # channel number information
  def __init__( self, ch=None, station=None,
                x=0., y=0., z=0., xname='lat', yname='lon',
                zname='elev', comp='U', bit=24., gain=0., unit=None, 
                period = 1., damp=0., amp=0., adc=0. ) :


    self.ch     = ch
    self.station = station
    self.x      = x
    self.y      = y
    self.z      = z
    self.xname  = xname
    self.yname  = yname
    self.zname  = zname
    self.comp   = comp
    self.bit    = bit
    self.gain   = gain
    self.unit   = unit
    self.period = period
    self.damp   = damp
    self.amp    = amp
    self.adc    = adc

class rn_chs( object ) :
  def __init__ ( self, n=1, chs = None ) :
    self.n     = n
    self.fdir  = ''
    self.fname = ''
    if chs is None :
      self.d = OrderedDict()
    else :
      self.d = OrderedDict()
      for ch in chs :
        self.d[ ch ] =  rn_ch( ch=ch ) 

  def set( self,  
              ch=None, station=None,  x=0., y=0., z=0., xname='lat', yname='lon',
              zname='elev',  comp='U', bit=24., gain=0., unit=None,
              period=1., damp=0., amp=0., adc=0. ) :
    if type( ch ) is str :
      self.d[ ch ] = rn_ch( ch=ch, station=station,x=x, y=y, z=z, xname=xname, yname=yname,
                          zname=zname, comp=comp, bit=bit, gain=gain, unit=unit,
                          period=period, damp=damp, amp=amp, adc=adc )
    else :
      self.d[ ch.ch ] = ch

    self.n = len( self.d.keys() )

  def remove( self, chs ) :
    if type(chs) is str :
      chs = [ chs ]

    nch = len( chs )
    for ch in chs :
      del self.d[ ch ]

    self.n -= nch



  def read( self, fname=None ) :
    self.d = OrderedDict()
    if fname is not None :
      self.fdir = os.path.dirname( fname )
      self.fname = os.path.basename( fname )

    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()

    self.n = len( lines )

    for line in lines :
      ( ch, station, x, y, z, xname, yname, zname, comp, unit, period,  bit, gain, 
          damp, amp, adc ) = line.split() 
      x = float(x)
      y = float(y)
      z = float(z)
      bit = float(bit)
      gain = float(gain) 
      period = float(period)
      damp   = float(damp)
      amp    = float(amp)
      adc    = float(adc)
      self.set( ch=ch, station=station,
                x=x, y=y, z=z, xname=xname, yname=yname, zname=zname, 
                comp=comp, bit=bit, gain=gain, unit=unit, period=period,
                damp=damp, amp=amp, adc=adc )

  def write( self, fname=None ) :
    if fname :
      self.fdir = os.path.dirname( fname )
      self.fname = os.path.basename( fname )

    outlines = []
    chs = self.d.keys()

    for ch in chs :
      chinfo = self.d[ ch ]
      outlines.append( '%s %s %f %f %f %s %s %s %s %s %f %f %e %e %e %e' % 
                       ( chinfo.ch, chinfo.station, chinfo.x, chinfo.y, chinfo.z, 
                       chinfo.xname, chinfo.yname, chinfo.zname, 
                       chinfo.comp, chinfo.unit, chinfo.period,  chinfo.bit,
                       chinfo.gain, 
                       chinfo.damp, chinfo.amp, chinfo.adc ) )

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


    self.t0    = datetime.datetime.today()
    self.t1    = datetime.datetime.today()
    self.data    = np.zeros( ( self.nchannels, self.samp_freq) )
    self.totalsec  = 0
    self.nsamples  = 0
    self.data    = None

    self.chs = rn_chs()
    self.channels = []
  

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

    #self.chs = rn_chs( n=self.nchannels )
    self.chs = rn_chs()
    for ch in self.channels :
      self.chs.set( ch=ch )
    #self.chs.id = self.channels
    #self.chs.x  = np.zeros( self.chs.n, dtype=np.float )
    #self.chs.y  = np.zeros( self.chs.n, dtype=np.float )
    #self.chs.z  = np.zeros( self.chs.n, dtype=np.float )
    print 'channels: ', self.channels

    # now find start time 
    date0 = info[ self.nchannels+1 ].split()[2]
    time0 = info[ self.nchannels+1 ].split()[3]
    year = int( date0[0:2] ) + 2000
    mon = int( date0[2:4] )
    day = int( date0[4:6]  )
    hh = int( time0[0:2] )
    mm = int( time0[2:4] )
    ss = int( time0[4:6]  )
    self.t0 = tz_jp.localize( datetime.datetime( year, mon, day, hh, mm, ss ) )
    print 'start time', self.t0

    # now find end time
    date1 = info[ -4 ].split()[2]
    time1 = info[ -4 ].split()[3]
    year = int( date1[0:2] ) + 2000
    mon = int( date1[2:4] )
    day = int( date1[4:6]  )
    hh = int( time1[0:2] )
    mm = int( time1[2:4] )
    ss = int( time1[4:6]  )
    self.t1 = tz_jp.localize( datetime.datetime( year, mon, day, hh, mm, ss ) )
    print 'end time: ', self.t0  

  # this is left for some historical reason
  def read_header_nofbin(self, fheader=None):

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


  def read_header(self, fheader=None, sett='n' ):
    if fheader is not None :
      self.fheader = fheader


    if os.path.dirname( self.fheader ) :
      self.fdir = os.path.dirname( self.fheader )

    self.chs.fdir = self.fdir

    self.fheader = os.path.basename( self.fheader )
    with open( os.path.join( self.fdir, self.fheader) ) as f:

      # line 1 : fbin
      self.fbin = f.readline().rstrip()

      # line 2 : sampling frequency
      self.samp_freq = int( float( f.readline() ) )

      # line 3 : staring time
      tmp = f.readline().split()
      try :
        (year0, mon0, mday0, yday0, hour0, min0, sec0,  microsec0 ) = [  int(x) 
                        for x in tmp  ]
      except :
        (year0, mon0, mday0, yday0, hour0, min0, sec0) = [  int(x) 
                        for x in tmp ]
        microsec0 = 0

      self.t0 = tz_jp.localize(
          datetime.datetime(year0, mon0, mday0, hour0, min0, sec0, microsec0 ) )
      self.yday0 = yday0

      # line 4 : end time
      tmp = f.readline().split()
      try :
        (year1, mon1, mday1, yday1, hour1, min1, sec1, microsec1) = [  int(x) 
                        for x in tmp ]
      except :
        (year1, mon1, mday1, yday1, hour1, min1, sec1) = [  int(x) 
                        for x in tmp ]
        microsec1 = 0


      self.t1 = tz_jp.localize(
          datetime.datetime(year1, mon1, mday1, hour1, min1, sec1, microsec1 ) )
      self.yday1 = yday1

      # line 5 : # of channels and channel file
      ( self.nchannels, self.chs.fname ) = f.readline().rstrip().split()
      self.chs.fname = self.chs.fname.rstrip()
      self.nchannels = int( self.nchannels )
      self.chs.n = self.nchannels 
      self.chs.read( )

      # line 6 : channel names ( order of data )
      self.channels = f.readline().split()

      self.set_totalsec_nsamples()

    if sett == 'y' :
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
      self.chs.fdir = self.fdir


    self.set_totalsec_nsamples()
    t0 = self.t0.timetuple()
    t1 = self.t1.timetuple()

    outline=[]
    outline.append('%s'%self.fbin)
    outline.append('%f'%self.samp_freq)
    outline.append('%4d %02d %02d %03d %02d %02d %02d %06d'%( 
            t0.tm_year, t0.tm_mon, t0.tm_mday, t0.tm_yday, 
            t0.tm_hour, t0.tm_min, t0.tm_sec, self.t0.microsecond ))
    outline.append('%4d %02d %02d %03d %02d %02d %02d %06d'%( 
            t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_yday, 
            t1.tm_hour, t1.tm_min, t1.tm_sec, self.t1.microsecond ))
    outline.append('%d %s'%(self.chs.n, self.chs.fname ))
    outline.append(' '.join(self.chs.d.keys())) #
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
           ).reshape(self.chs.n, self.nsamples)

  def check_fsize( self ) :
    fbin = os.path.join( self.fdir, self.fbin ) 
    print( 'Checking binary file size: %s'%fbin )
    try :
      statinfo = os.stat( fbin ) 
      fsize = statinfo.st_size
      fsize0 = self.nsamples*self.chs.n*4
      print( self.chs.n )
      if fsize == fsize0 :
        print( 'Binary file size ok. %d bytes'%( fsize) )
      else :
        print( 'WARNING: Incorrect binary file size\n %d bytes (expected) vs %d bytes (actual) '%( fsize0, fsize) )
    except :
      print( 'WARNING: Something wrong with %s!'%fbin )


  


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


  # write all data
  def write_data( self, fbin=None ) :
    if fbin is not None :
      self.fbin = fbin

    self.data.astype('float32').tofile( os.path.join( self.fdir, self.fbin ) )   
  # write data append
  def open_data( self, op='w' ): #{{{{{
    self.fbinh = open( os.path.join( self.fdir, self.fbin ), op+'b' )

  def write_data_ch( self, ich ) :
    if not self.fbinh :
      self.open_data()
    self.fbinh.seek( 4 * ich * self.nsamples, os.SEEK_SET ) 
    self.data.astype( np.float32).tofile( self.fbinh )

  def close_data( self ) :
    self.fbinh.close()


    
