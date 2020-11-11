#!/usr/bin/env python
import datetime
import numpy as np
import pytz 
from dateutil import parser

import oshima.lib as rk_lib


tz_jp = pytz.timezone('Asia/Tokyo')

class rk_binary(object) :
  def __init__(self):
    self.samp_freq = 250
    self.nchannels = 4
    self.channels  = []
    self.t0    = datetime.datetime.today()
    self.t1    = datetime.datetime.today()
    self.data    = np.zeros( ( self.nchannels, self.samp_freq) )
    self.totalsec  = 0
    self.nsamples  = 0
    self.data    = None

  def initialize(self, val = 0. ) :
    self.totalsec = int( (self.t1 - self.t0 ).total_seconds() ) + 1 
    self.nsamples = int( self.samp_freq * self.totalsec )
    self.data  = np.ones( ( self.nchannels, self.nsamples ) ,
                dtype = 'float_') * val


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
#    self.t0 = datetime.datetime.strptime(date0+time0,'%y%m%d%H%M%S')
    self.t0 = parser.parse(data0+time0+'+0900')
    print 'start time: ', date0, time0

    # now find end time
    date1 = info[ -4 ].split()[2]
    time1 = info[ -4 ].split()[3]
    #self.t1  = datetime.datetime.strptime(date1+time1,'%y%m%d%H%M%S')
    self.t1 = parser.parse(data1+time1+'+0900')
    
    print 'end time: ', date1, time1

  def read_header(self, fheader):
    with open(fheader) as f:
      self.samp_freq = int( float( f.readline() ) )
      (year0, mon0, mday0, yday0, hour0, min0, sec0) = [  int(x) 
                        for x in f.readline().split() ]
      self.t0 = datetime.datetime(year0, mon0, mday0, hour0, min0, sec0,
            tzinfo = tz_jp )
      self.yday0 = yday0
      (year1, mon1, mday1, yday1, hour1, min1, sec1) = [  int(x) 
                        for x in f.readline().split() ]
      self.t1 = datetime.datetime(year1, mon1, mday1, hour1, min1, sec1,
            tzinfo = tz_jp)
      self.yday1 = yday1
      self.nchannels = int(f.readline())
      self.channels  = f.readline().split()

      self.set_totalsec_nsamples()


  def set_totalsec_nsamples( self ) :
    self.totalsec = int( (self.t1 - self.t0 ).total_seconds() ) + 1 
    self.nsamples = self.samp_freq * self.totalsec

  def set_t( self ) :
    self.t = []
    for it in range( self.nsamples ) :
      self.t.append( self.t0 + datetime.timedelta( 0, float(it) / self.samp_freq ) )



  def write_header(self, fheader):
    self.set_totalsec_nsamples()
    t0 = self.t0.timetuple()
    t1 = self.t1.timetuple()

    outline=[]
    outline.append('%f'%self.samp_freq)
    outline.append('%4d %02d %02d %03d %02d %02d %02d'%( 
            t0.tm_year, t0.tm_mon, t0.tm_mday, t0.tm_yday, 
            t0.tm_hour, t0.tm_min, t0.tm_sec))
    outline.append('%4d %02d %02d %03d %02d %02d %02d'%( 
            t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_yday, 
            t1.tm_hour, t1.tm_min, t1.tm_sec))
    outline.append('%d'%self.nchannels)
    outline.append(' '.join(self.channels))
    outline.append('%d %f'%( self.nsamples, self.totalsec ))
    with open(fheader,'w') as f:
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
  def read_data_int(self, fbin):
    self.totalsec = int( (self.t1 - self.t0 ).total_seconds() ) + 1 
    self.nsamples = self.samp_freq * self.totalsec
    print self.nsamples,self.nchannels
    self.data = np.fromfile( fbin, dtype = np.dtype('i4') 
           ).reshape(self.nchannels, self.nsamples)

  # use this if data is binary
  def read_data( self, fbin ) :
    self.set_totalsec_nsamples() 
    print self.nsamples,self.nchannels
    self.data = np.fromfile( fbin, dtype = np.dtype('float32') 
           ).reshape(self.nchannels, self.nsamples)

  def extract_time( self, t0, t1 ) :
    self.set_t()
    
    if t0 > self.t0 :
      it0 = self.t.index( t0 ) #int( ( t0 - self.t0 ).total_seconds() * self.samp_freq ) 
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
  def write_data( self, fbin ) :
    self.data.astype('float32').tofile( fbin )   


  #======================================================================
  # some processing
  #======================================================================

  def normalize( self ) :
    self.data = rk_lib.normalize( self.data )

  def bandpass( self, lowcut, highcut, order=5 ) :
    self.data = rk_lib.butter_bandpass_filter( 
                                  self.data, lowcut, highcut, 
                                  float(self.samp_freq), order=order)

  def rfft( self ): 
    self.data  = rk_lib.rk_hanning_taper( self.data, 10 )
    self.fdata = rk_lib.rk_rfft( self.data )
    self.f     = np.arange( 0., self.nsamples/2 + 1, dtype=np.float 
                          )  / float( self.nsamples ) * self.samp_freq
