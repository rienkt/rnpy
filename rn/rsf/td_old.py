#!/usr/bin/env python

# Import RSF structure (like SEP.top)
import rsf.api as rsf
import sys
import os.path
# Import math packages
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg') # this will disable X connection
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from math import sqrt

# define function

def get_dim(input):
  dim=0
  for i in range(1,10):
    if input.int('n'+str(i))==None:
      break
    else:
      dim=i
  return dim
 
     
class TimedomainData:
  def __init__(self, ot  =0., dt  =1., nt  =1,
                     orcv=0., drcv=1., nrcv=1,
                     osrc=0., dsrc=1., nsrc=1,
                     scale=1., tscale=1.):
    self.ot     = ot
    self.dt     = dt
    self.nt     = nt
    self.orcv   = orcv
    self.drcv   = drcv
    self.nrcv   = nrcv
    self.osrc   = osrc
    self.dsrc   = dsrc
    self.nsrc   = nsrc
    self.data   = np.zeros( ( self.nsrc, self.nrcv, self.nt ), 'f' )
    self.scale  = scale
    self.tscale = tscale

    self.unit1  = None
    self.unit2  = None
    self.label1 = None
    self.label2 = None

  def read_header( self, input=None, finput=None, sfile='src.dat', rfile='rcv.dat' ):
    if input is None :        
      input = rsf.Input(finput)

    self.dim=get_dim(input)
   
    print input.int('n1')
 
    self.nt     = input.int('n1')
    self.nrcv   = input.int('n2')
    self.dt     = input.float('d1')*self.tscale
    self.drcv   = input.float('d2')
    self.ot     = input.float('o1')*self.tscale
    self.orcv   = input.float('o2')
    self.unit1  = input.string('unit1')
    self.unit2  = input.string('unit2')
    self.label1 = input.string('label1')
    self.label2 = input.string('label2')

    print self.nt

    if self.dim >= 3:
      print "dim >= 3"
      self.nsrc = input.int('n3')
      self.dsrc = input.float('d3')
      self.osrc = input.float('o3')
    else:
      print "dim =1 or 2"
      self.nsrc = 1
      self.dsrc = 1.
      self.osrc = 0.

    try:
      self.srcx = np.fromfile( sfile, count = self.nsrc * 2, sep = ' ' 
                             ).reshape( [ self.nsrc, 2 ] )[ :, 0 ]
      self.srcz = np.fromfile( sfile, count = self.nsrc * 2, sep = ' ' 
                             ).reshape( [ self.nsrc, 2 ] )[ :, 1 ]
      self.srcx *= self.scale
      self.srcz *= self.scale
    except:
      self.srcx = np.arange( 0, self.nsrc, dtype=np.float ) * self.dsrc + self.osrc
      self.srcz = np.arange( 0, self.nsrc, dtype=np.float ) * self.dsrc + self.osrc
      print 'source coordinate file does not exist'


    try:
      self.rcvx = np.fromfile( rfile, sep=' ' ).reshape( [self.nrcv, 2] )[ :, 0 ]
      self.rcvz = np.fromfile( rfile, sep=' ' ).reshape( [self.nrcv, 2] )[ :, 1 ]
      self.rcvx = self.rcvx * self.scale
      self.rcvz = self.rcvz * self.scale
    except:
      self.rcvx = np.arange( 0, self.nrcv, dtype=np.float ) * self.drcv + self.orcv
      self.rcvz = np.arange( 0, self.nrcv, dtype=np.float ) * self.drcv + self.orcv
      print 'receiver coordinate file does not exist'
      
    self.t = np.arange( 0, self.nt, dtype=np.float ) * self.dt + self.ot

    self.s = np.arange( 0, self.nsrc, dtype=np.float ) * self.dsrc + self.osrc
    self.r = np.arange( 0, self.nrcv, dtype=np.float ) * self.drcv + self.orcv
    # we are only considering meter for this case
    try :
      self.s = self.s * self.scale
      self.r = self.r * self.scale
    except:
      print 'no scaling factor is given for source/receiver coordinate'
  

  def read_rsf( self, input=None, finput=None, sfile='src.dat', rfile='rcv.dat' ):
    if input is None :    
      input=rsf.Input(finput)

    self.read_header( input=input, sfile=sfile, rfile=rfile)
    self.data=np.zeros((self.nsrc,self.nrcv,self.nt),'f')
    input.read(self.data)
#    input.close()
  


  def write_rsf(self,output):
    output.put('n1',self.nt)
    output.put('n2',self.nrcv)
    output.put('n3',self.nsrc)
    output.put('d1',self.dt)
    output.put('d2',self.drcv)
    output.put('d3',self.dsrc)
    output.put('o1',self.ot)
    output.put('o2',self.orcv)
    output.put('o3',self.osrc)
    output.write(np.float32(self.data))
    output.close()

  def normalise(self,tw1=0.,tw2=0.,isrc1=1,isrc2=1): # source number starts from 1
    if tw2==0 :
      tw2=self.ot+(self.nt-1)*self.dt
    if isrc2==0:
      isrc2=self.nsrc
    # calculate it0 and it1
    itw0=np.maximum(np.round((tw1-self.ot)/self.dt).astype('int'),0)
    itw1=np.minimum(np.round((tw2-self.ot)/self.dt).astype('int'),self.nt-1)
    self.nw=itw1-itw0+1
    self.tw1=tw1
    self.tw2=tw2
    nsrcw=isrc2-isrc1
    print itw0,itw1,self.nw, isrc1,isrc2,nsw,self.nr
    print self.data[(isrc1-1):isrc2,:, itw0:(itw1+1)].shape,self.nw
    # data [ nt, nr, ns ]
    self.data2=self.data[(isrc1-1):isrc2,:,itw0:(itw1+1)].reshape((1,self.nw*self.nrcv*nsrcw))
    amp_max=np.amax(np.abs(self.data2))
    amp_rms=np.sqrt(np.sum(np.square(np.abs(self.data2)))*self.dt
        /(self.nw*self.nrcv*self.nsrc))
    print amp_max, amp_rms
    self.data=self.data/amp_rms


     
class TimedomainData_draw(TimedomainData):
  def format_figure( self, ax, par ):
    ax.set_xlabel( par.xlabel,fontsize=18 )
    ax.set_ylabel( par.ylabel,fontsize=18 )
    ax.set_xlim( par.xmin, par.xmax )
    ax.set_ylim( par.ymax, par.ymin )
    ax.xaxis.set_label_position( 'top' )
    ax.xaxis.set_ticks_position( 'top' )

    if par.xtick :
      ax.set_xticks( np.arange( par.xmin, par.xmax + par.xtick, par.xtick ) )
    if par.ytick :
      ax.set_yticks( np.arange(par.ymin,par.ymax+par.ytick,par.ytick))
    pl.tick_params( axis='both', which='major', labelsize=18)
    pl.tick_params( axis='both', which='minor', labelsize=18)


    # add text if given
    #print epar.text,epar.textx,epar.texty
    if par.text :
      ax.text( par.textx, par.texty, par.text, fontsize=18, color='white')

    # set aspect ratio
    try: 
      print par.aspect, ( ax.get_xlim()[1] - ax.get_xlim()[0] ),  ( 
                          ax.get_ylim()[1] - ax.get_ylim()[0] )
      # next two lines avoid some silly Unicode conversion errors
      myaspect = par.aspect * abs( ( ax.get_xlim()[1] - ax.get_xlim()[0] ) / ( 
                                     ax.get_ylim()[1] - ax.get_ylim()[0] ) )
      ax.set_aspect( float( myaspect ) )
    except:
      print 'aspect ratio is not defined'



  def draw( self, par, save='y' ):
    data = par.set_axis_range(self)
    fig = pl.figure()
    ax  = fig.add_subplot(111, adjustable='box')


    im = ax.imshow( np.flipud(data.transpose()), interpolation='spline16',
                    extent = ( par.xmin, par.xmax, par.t[0], par.t[-1] ),
                    cmap = par.cm, rasterized = 'True')
    im.set_norm(matplotlib.colors.Normalize(vmin=par.vmin,vmax=par.vmax))

    ax.images.append(im)
    self.format_figure(ax,par)

    if save == 'y' :

      fig.savefig(par.imfile+'.png',bbox_inches='tight',dpi=300)
      fig.savefig(par.imfile+'.pdf',bbox_inches='tight',dpi=300)

    return fig


  def wiggle(self,par):
    data=par.set_axis_range(self)
    print data.shape


    fig = pl.figure()
    ax = fig.add_subplot(111)        
    # scales the trace amplitudes relative to the number of traces
    scalar=par.wiggle_scale/np.max(np.abs(data.ravel()))
    # set the very last value to nan. this is a lazy way to prevent wrapping
    print 'scalar',scalar  
    data[:,-1] = np.nan
    vals = data.ravel() #flat view of the 2d array.

    # flat index array, for correctly location zero crossing in the flat view
    vect = np.arange(vals.size).astype(np.float) 

    # index before zero crossing
    crossing = np.where(np.diff(np.signbit(vals)))[0] 
    crossing = np.delete(crossing, np.where(np.isnan(vals[crossing+1])))

    # use linear interpolation to find the zero crossing, i.e. y = mx + c. 
    x1=  vals[crossing]
    x2 =  vals[crossing+1]
    y1 = vect[crossing]
    y2 = vect[crossing+1]
    m = (y2 - y1)/(x2-x1)
    c = y1 - m*x1       
    print 'x1',x1,'x2',x2,'x2-x1',x2-x1,'y2-y1',y2-y1,'m',m,'c',c
    #tack these values onto the end of the existing data
    x = np.hstack([vals, np.zeros_like(c)])
    y = np.hstack([vect, c])
    print vals
    print np.where(x<0)
    #resort the data
    order = np.argsort(y) 
    #shift from amplitudes to plotting coordinates
    x_shift, y = y[order].__divmod__(self.nt)
  
    # now change the coordinate for y as time axis
    y=y*self.dt + self.ot

    print x,y,self.dt,self.ot

    ax.plot(x[order] *scalar + x_shift + 1 + par.x[0], y, 'k')
    x[x<0] = np.nan
    x = x[order] *scalar + x_shift + 1 + par.x[0]


    ax.fill(x,y, 'k', aa=True)

    self.format_figure(ax,par)

    fig.savefig(par.imfile+'.png',bbox_inches='tight',dpi=300)

    return fig


  def image_wiggle(self,par):
    data=par.set_axis_range(self)
    print data.shape
    # calibrate for image
    data_im=np.vstack((np.zeros((1,self.nt),'f'), data, np.zeros((1,self.nt),'f')))
    im_xmin=par.x[0]-(par.x[1]-par.x[0])
    im_xmax=par.x[-1]+par.x[-1]-par.x[-2]
    fig = pl.figure()
    ax = fig.add_subplot(111)        
    im=ax.imshow(np.flipud(data_im.transpose()),interpolation='spline16',
      extent=(im_xmin,im_xmax,par.t[0],par.t[-1]),cmap=par.cm,rasterized='True')
    im.set_norm(matplotlib.colors.Normalize(vmin=par.vmin,vmax=par.vmax))

    ax.images.append(im)

  # scales the trace amplitudes relative to the number of traces
    scalar=par.wiggle_scale/np.max(np.abs(data.ravel()))
    # set the very last value to nan. this is a lazy way to prevent wrapping
    print 'scalar',scalar  
    data[:,-1] = np.nan
    vals = data.ravel() #flat view of the 2d array.

    # flat index array, for correctly location zero crossing in the flat view
    vect = np.arange(vals.size).astype(np.float) 

    # index before zero crossing
    crossing = np.where(np.diff(np.signbit(vals)))[0] 
    crossing = np.delete(crossing, np.where(np.isnan(vals[crossing+1])))

    # use linear interpolation to find the zero crossing, i.e. y = mx + c. 
    x1=  vals[crossing]
    x2 =  vals[crossing+1]
    y1 = vect[crossing]
    y2 = vect[crossing+1]
    m = (y2 - y1)/(x2-x1)
    c = y1 - m*x1       
    print 'x1',x1,'x2',x2,'x2-x1',x2-x1,'y2-y1',y2-y1,'m',m,'c',c
    #tack these values onto the end of the existing data
    x = np.hstack([vals, np.zeros_like(c)])
    y = np.hstack([vect, c])
    print vals
    print np.where(x<0)
    #resort the data
    order = np.argsort(y) 
    #shift from amplitudes to plotting coordinates
    x_shift, y = y[order].__divmod__(self.nt)
  
    # now change the coordinate for y as time axis
    y=y*self.dt + self.ot

    print x,y,self.dt,self.ot
    xorig = x[ order ]
    x = xorig * scalar + x_shift + par.x[0]
    ax.plot(x, y, 'k')
    x[xorig < 0 ] = np.nan


    ax.fill(x,y, 'k', aa=True)

    self.format_figure(ax,par)

    fig.savefig(par.imfile+'.png',bbox_inches='tight',dpi=300)




  def prep_crosswell(self,par):      
    if par.xaxis=='z':
       self.r=self.rz
    elif par.xaxis=='x':
      self.r=self.rz
    par.xaxis=None

    tmp=copy.copy(self.r)
    self.r=copy.copy(self.t)
    self.t=copy.copy(tmp)

    self.nr=self.r.shape[0]
    self.nt=self.t.shape[0]




    tmp1=copy.copy(epar.xmin)
    tmp2=copy.copy(epar.xmax)
    epar.xmin=copy.copy(epar.ymin)
    epar.xmax=copy.copy(epar.ymax)
    epar.ymin=copy.copy(tmp1)
    epar.ymax=copy.copy(tmp2)
    tmp1=copy.copy(epar.xtick)
    epar.xtick=epar.ytick
    epar.ytick=tmp1
    print 'check', epar.xmin, epar.xmax, epar.ymin, epar.ymax,epar.xtick,epar.ytick
    tmp=copy.copy(epar.xlabel)
    self.data=np.transpose(self.data,(1,2,0))





class draw_par:
  def __init__(self):
    self.fname   = ''
    self.imfile  = 'ttt'
    self.clip    = None; self.bias    = None
    self.xmin    = None; self.xmax    = None
    self.ymin    = None; self.ymax    = None
    self.cmin    = None; self.cmax    = None
    self.vmin    = None; self.vmax    = None
    self.cmap    = None
    self.cm      = None

    self.xlabel  = 'X'; self.ylabel = 'Y';
  
    self.gather = None; 
    self.data_xmin = 0; self.data_xmax = 0.;
    self.data_tmin = 0; self.data_tmax = 0.;
    self.x = None; self.t = None; 

    self.xtick = None; self.ytick = None;

    self.text = None; 
    
    self.aspect = None;


  def set_crange( self, cmin=None, cmax=None ):
    if cmin :
      self.cmin = cmin
    if cmax :
      self.cmax = cmax



    if self.cmin :
      self.vmin = self.cmin
      self.vmax = self.cmax
    else :
      self.vmin = self.bias - self.clip
      self.vmax = self.bias + self.clip

  def set_cmap( self, cmap=None ):
    if cmap :
      self.cmap = cmap
    if self.cmap == 'e' :
      self.cmap = 'seismic_r'
      self.cm = pl.cm.seismic_r
    elif self.cmap =='j' :
      self.cmap = 'jet'
      self.cm = pl.cm.jet
    else :
      self.cmap = 'binary' 
      self.cm = pl.cm.binary

  def set_label(self, w, xlabel=None, ylabel=None ):
    if xlabel:
      self.xlabel = xlabel
    elif w.label2:
      self.xlabel = w.label2
      if w.unit2 :
        self.xlabel = w.label2 + ' ['+w.unit2+']'
    else:
      self.xlabel = 'Depth [m]'

    if ylabel : 
      self.ylabel = ylabel
    elif w.label1 :
      self.ylabel = w.label1
      if w.unit1 :
        self.ylabel = w.label1 + ' [' + w.unit1 + ']'
    else:
      self.ylabel = 'Time [ms]'

  def set_axis_range(self, td, gather=None, inum=None, xaxis=None):
    if gather :
      self.gather = gather

    if inum :
      self.inum = inum

    if xaxis :
      self.xaxis = xaxis

    if self.gather == 's':
      data = copy.copy( td.data[ self.inum - 1, :, : ] )
      if self.xaxis == 'z' :
        r = td.rcvz
      elif self.xaxis == 'x':
        r = td.rcvx
      self.x = r
      self.data_xmin = td.r[ 0 ]
      self.data_xmax = td.r[ td.nrcv - 1 ]



    # you need to prep when source/receiver is not available
    elif self.gather == 'r':
      data = copy.copy( td.data[ :, self.inum-1, : ] )
      if self.xaxis == 'z':
        s = td.srcz
      elif self.xaxis == 'x':
        s = td.srcx
      self.x = s
      self.data_xmin = s[0]
      self.data_xmax = s[td.nsrc-1]
    self.t = td.t
    self.data_tmin = td.t[0]
    self.data_tmax = td.t[td.nt-1]

    if self.xmin == None : 
      self.xmin = self.data_xmin
    if self.xmax == None:
      self.xmax = self.data_xmax
    if self.ymin == None :
      self.ymin = self.data_tmin
    if self.ymax == None : 
      self.ymax = self.data_tmax

    return data


  


