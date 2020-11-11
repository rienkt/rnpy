#!/usr/bin/env python
"""

  Plotting routines for continuous data 

  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np
from matplotlib import dates
import matplotlib.pylab as plt
from .core import rk_binary
import pytz

from oshima.plot.annotate import align_marker
from oshima.plot.format   import FigFormat
import oshima.lib as rk_lib

#======================================================================
# Parameters
#======================================================================

tz_jp = pytz.timezone('Asia/Tokyo')

#======================================================================
# classe / functions
#======================================================================
def add_time_marker( ax, times ) :
  for time_plot in times :
    plt.plot_date( dates.date2num( time_plot['time'] ), 0,
            marker=align_marker(time_plot['marker'], valign='bottom'),
            ms=12, clip_on=False, 
            markerfacecolor=time_plot['color'],
            markeredgecolor=time_plot['color'],
            markeredgewidth=0.)

def wiggle( ax, dfig, axpar, wiggle_scale=1., iskip=1, ichskip=1 ) :

#    print 'wiggle', dfig.data.shape
#    print len(dfig.t),dfig.data.shape
#    print 'set_t'
    dfig.set_t()
#    print len(dfig.t),dfig.data.shape
#    print 'tplot'
    tplot = dates.date2num( dfig.t[::iskip] )

    dplot = dfig.data[ ::ichskip, ::iskip ]
    nchannels = dplot.shape[0]  

#    print len(dfig.t), len(tplot), dfig.data.shape
  
  # scales the trace amplitudes relative to the number of traces
    amax = np.max( np.abs( dplot ) ) 
#    amax = 1.
    if amax > 1.0e-28  :
      scalar = wiggle_scale / amax * ichskip
    else :
      scalar = wiggle_scale * ichskip

    dplot *= scalar

    print dfig.nchannels, scalar

    xshift = np.arange( 0., dfig.nchannels, ichskip, dtype=np.float ) + 1.

    for ichannel in range( nchannels ) :
      dfig.data[ ichannel, : ] += xshift[ ichannel ] 

    print tplot[0], tplot[-1]
#    print dfig.data
#    print amax, dfig.data.shape

    # set axpar
    axpar.xmin  = tplot[0]
    axpar.xmax  = tplot[-1]
    axpar.ymin  = dfig.nchannels + 1.
    axpar.ymax  = 0.
#    axpar.title = '%s - %s ' % ( dfig.t0.strftime('%Y-%m-%d %H:%M:%S'),
#                    dfig.t1.strftime('%Y-%m-%d %H:%M:%S') )
    ax.plot_date( tplot,
                  dplot.transpose(), 
                   fmt='-',
                   color='k',  linewidth=0.1 , tz=tz_jp )
    axpar.format_figure( ax )

def spectra( ax, dfig, axpar, iskip=1, ichskip=1, norm=1 ) :

#    print dfig.fdata
#    print dfig.fdata.shape
#    print dfig.f[0], dfig.f[-1]
#    print dfig.nchannels

    idx = np.argmin( np.abs( dfig.f - axpar.xmax ) ) 

    if norm == 1 :
      fdata_plot = rk_lib.normalize( dfig.fdata[ ::ichskip, : idx +1 : iskip ] )
    else :
      fdata_plot = dfig.fdata[ ::ichskip, : idx +1 : iskip ] 

    # set axpar
    axpar.ymin  = dfig.nchannels + 1.
    axpar.ymax  = 0.
#    axpar.title = '%s - %s ' % ( dfig.t0.strftime('%Y-%m-%d %H:%M:%S'),
#                    dfig.t1.strftime('%Y-%m-%d %H:%M:%S') )

    ax.imshow( fdata_plot,  
                   extent=[ dfig.f[0], dfig.f[idx], 
                            dfig.nchannels+1, 1 ], 
                   aspect='auto' )
    axpar.format_figure( ax )



