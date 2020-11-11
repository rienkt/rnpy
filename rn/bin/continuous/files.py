#!/usr/bin/env python
"""
  FILES.PY
    misc files 


"""
import numpy as np
import datetime
import pytz
from dateutil import parser
tz_jp = pytz.timezone('Asia/Tokyo')


## shot information

#def read_shot_info( fname ) :
#  with open( fname ) as f :
#    info = f.read().splitlines()
#    
#  shots = []
#  for line in info :
#    ( myid, myyear, mymon, mymday, myhour, mymin, mysec, 
#      mylat, mylon, myelev ) = line.split()
#    shots.append( { 'id'   : int(myid),
#                    'time' : datetime.datetime( int(myyear), int(mymon),
#                             int(mymday), int(myhour),
#                             int(mymin),  int(mysec), 
#                             tzinfo = tz_jp ),
#                             'lat' : float(mylat), 'lon' : float(mylon), 
#                    'elev': float(myelev) } )
#  return shots

def write_shot_info( fname, dlist ) :
  outlines = []
  for d in dlist :
    # assume data is always GMT+9
    d[ 'outtime' ] = d[ 'time' ].strftime('%Y-%m-%d %H:%M:%S')+'+0900'
    outlines.append( '%(index)d %(id)s %(lat)f %(lon)f %(elev)f %(outtime)s'%d )
  
  with open( fname, 'w' ) as f :
    f.write( '\n'.join( outlines ) )

def write_rcv_info( fname, dlist ) :
  outlines = []
  for d in dlist :
    # assume data is always GMT+9
    d[ 'outtime0' ] = d[ 'time0' ].strftime('%Y-%m-%d %H:%M:%S')+'+0900'
    d[ 'outtime1' ] = d[ 'time1' ].strftime('%Y-%m-%d %H:%M:%S')+'+0900'
    outlines.append( '%(index)d %(id)s %(ch)s %(lat)f %(lon)f %(elev)f %(outtime0)s %(outtime1)s'%d )
  
  with open( fname, 'w' ) as f :
    f.write( '\n'.join( outlines ) ) 




def read_shot_info( fname ) :
  with open( fname ) as f :
    info = f.read().splitlines()
    
  shots = []
  for line in info :
    ( myindex, myid, mylat, mylon, myelev, myday, myhour ) = line.split()
    shots.append( { 'index': int(myindex),
                    'id'   : myid,
                    'time' : parser.parse( myday + 'T' + myhour ),
                    'lat' : float(mylat), 'lon' : float(mylon), 
                    'elev': float(myelev) } )
  return shots



### receiver infromation
def read_rcv_info( fname ) :
  with open( fname ) as f :
    info = f.read().splitlines()
    
  rcvs = []
  for line in info :
    ( myindex, mych, myid, mylat, mylon, myelev, 
              myday0, myhour0, myday1, myhour1) = line.split()
    rcvs.append( {  'index': int(myindex), 
                    'ch'   : mych,
                    'id'   : myid,
                    'time0' : parser.parse( myday0 + 'T' + myhour0 ),
                    'time1' : parser.parse( myday1 + 'T' + myhour1 ),
                    'lat' : float(mylat), 'lon' : float(mylon), 
                    'elev': float(myelev) } )
  return rcvs




