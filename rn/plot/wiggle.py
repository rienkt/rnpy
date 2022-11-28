#!/usr/bin/env python
"""

  WIGGLE.py
  draw wiggle


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np

import matplotlib.pyplot as plt
import copy

#======================================================================
# classes / functions
#======================================================================


def wiggle( ax, datain, xax, yax, wiggle_scale=1., direction='v', fill='y' ):


  data = copy.copy( datain )

  dx = xax[1]-xax[0]
  nx = len(xax)
  ox = xax[0]
  dy = yax[1]-yax[0]
  ny = len(yax)
  oy = yax[0]




  # scales the trace amplitudes relative to the number of traces

  scalar = wiggle_scale / np.max(np.abs(data.ravel()))

  # set the very last value to nan. this is a lazy way to prevent wrapping
  print( 'scalar',scalar   )
  data[:,-1] = np.nan
  vals = data.ravel() #flat view of the 2d array.

  # flat index array, for correctly location zero crossing in the flat view
  vect = np.arange(vals.size).astype(np.float) 

  # index before zero crossing
  crossing = np.where(np.diff(np.signbit(vals)))[0] 
  crossing = np.delete(crossing, np.where(np.isnan(vals[crossing+1])))

  # use linear interpolation to find the zero crossing, i.e. y = mx + c. 
  x1 = vals[crossing]
  x2 = vals[crossing+1]
  y1 = vect[crossing]
  y2 = vect[crossing+1]
  m = (y2 - y1)/(x2-x1)
  c = y1 - m*x1       
  #print 'x1',x1,'x2',x2,'x2-x1',x2-x1,'y2-y1',y2-y1,'m',m,'c',c
  #tack these values onto the end of the existing data
  x = np.hstack([vals, np.zeros_like(c)])
  y = np.hstack([vect, c])
  #print vals
  #print np.where(x<0)

  #resort the data
  order = np.argsort(y) 
  #shift from amplitudes to plotting coordinates
  x_shift, y = y[order].__divmod__( ny )

  # now change the coordinate for y as time axis
  y = y*dy + oy

  if direction == 'v' :
    ax.plot( x[order] *scalar + x_shift + 1 + ox, y, 'k')
  else :
    ax.plot( y, x[order] *scalar + x_shift + 1 + ox, 'k')


  if fill == 'y' :
    x[x<0] = np.nan
    x = x[order] *scalar + x_shift + 1 + ox


    ax.fill(x,y, 'k', aa=True)


def wiggle_simple( ax, din, x, y, 
      wiggle_scale=1., direction='v', norm='y', linewidth=0.5,   
      linecolors='k', linealphas = 1.0, linestyles = '-' ) :



  n0, n1 = din.shape

  dx = x[1] - x[0]
  dy = y[1] - y[0]

  if linecolors is None :
    linecolors =[ 'k' for i in range(n0) ] 
  elif type(linecolors) == str :
    lcolor = linecolors
    linecolors = [ lcolor for i in range(n0) ]

  if linealphas is None :
    linealphas =[ 1.0 for i in range(n0) ] 
  elif type(linealphas) == float :
    lalpha = linealphas
    linealphas = [ lalpha for i in range(n0) ]

  if linestyles is None :
    linestyles =[ '-' for i in range(n0) ] 
  elif type(linestyles) == str :
    lstyle = linestyles
    linestyles = [ lstyle for i in range(n0) ]
  #print( linecolors )

  if norm == 'y' :
    scale = wiggle_scale / np.max( np.abs( din ) )
  else :
    scale = wiggle_scale
  #print( 'scale', scale, 'n0', n0 )

  if direction == 'h' :
    shift = np.arange( 0, n0, dtype=np.float ) * dy
    for i0 in range( n0 ) :
      ax.plot( x, din[ i0, : ] * scale + shift[ i0 ],  
          linewidth=linewidth, linestyle=linestyles[i0],
          color=linecolors[i0],
          alpha = linealphas[i0] )

  else :
    shift = np.arange( 0, n0, dtype=np.float ) * dx
    for i0 in range( n0 ) :
      ax.plot( din[ i0, : ] * scale + shift[ i0 ], y, color=linecolors[i0] )










  


