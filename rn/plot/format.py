import numpy as np 
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)
import matplotlib

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['pdf.fonttype'] = 42

import matplotlib.ticker as mticker

try :
  from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
  import cartopy.crs as ccrs

except :
  print( 'cartopy is not available' )
  #rcParams['png.fonttype'] = 42


def calc_aspect_from_axis_range( ax ) :
  return abs( ax.xmax - ax.xmin) / abs( ax.ymax - ax.ymin )

def calc_figsize_widths_heights( fig_paperwidth, width_ratios, height_ratios, 
                                 left, right, bottom, top, dwidth, dheight, 
                                 aspect, ax00_paperwidth=0, dwidths=None,
                                 dheights=None):
  #{{{{{
  print( aspect )
  nfig_hor = len( width_ratios )
  nfig_ver = len( height_ratios )

  width_ratios = np.asarray( width_ratios, dtype=np.float )
  height_ratios = np.asarray( height_ratios, dtype=np.float )
  if dwidths is None :
    dwidths = np.ones( nfig_hor -1 , dtype=np.float ) * dwidth
  else :
    dwidths = np.asarray( dwidths, dtype=np.float )
  #print( nfig_hor, dwidths )

  widths = ( ( 1. - left - right - np.sum(dwidths) ) *
             width_ratios / np.sum( width_ratios ) )
  if dheights is None :
    dheights = np.ones( nfig_ver - 1, dtype=np.float ) * dheight
    #print( nfig_ver, dheights )
  else :
    dheights = np.asarray( dheights, dtype=np.float )
    #heights = ( ( 1. - top - bottom - dheight*float( nfig_ver -1) ) *
    #           height_ratios / np.sum( height_ratios ) )
  heights = ( ( 1. - top - bottom - np.sum(dheights) ) *
               height_ratios / np.sum( height_ratios ) )
  if ax00_paperwidth > 0 :
    pwidth  = ax00_paperwidth / widths[0]
    pwidths = pwidth * widths
  else :
    pwidth =  fig_paperwidth
    pwidths = widths * pwidth

  pheight = pwidths[0] / aspect / heights[0]

  figsize = ( pwidth, pheight )


  return figsize, widths, heights

#}}}}}

#def create_colorbar( fig, axim, 
#    bottom=None, height=None, left=None, width=None,
#    dbottom=None, dleft=None, flag='h', relative='y') :
def create_colorbar( fig, axim=None, caxfmt=None,
    bottom=None, height=None, left=None, width=None,
    dbottom=None, dleft=None,
    flag='h' ) :
  #{{{{{

  if caxfmt :
    height = caxfmt.height
    width = caxfmt.width
    dleft = caxfmt.dleft
    dbottom = caxfmt.dbottom
    left = caxfmt.left
    bottom = caxfmt.bottom
    if caxfmt.orientation == 'vertical' :
      flag = 'v'

  if axim :
    impos = np.asarray( axim.get_position().bounds )
    axheight = impos[3]
    axwidth  = impos[2]
    if bottom :
      bottom *= axheight
    if dbottom :
      dbottom *= axheight
    if height :
      height *= axheight
    if dleft : 
      dleft *= axwidth
    if left :
      left *= axwidth
    if width :
      width *= axwidth
  else :
    impos = np.asarray( [ 0., 0., 1., 1. ] )


  if flag == 'h'  :
    impos[3] = height 
    if bottom :
      impos[1] = bottom 
    if dbottom :
      impos[1] += dbottom
    if width :
      impos[2] = width 
    if left :
      impos[0] += left
  else :
    print( dleft, left )
    if dleft is not None:
      impos[0] += ( impos[2] + dleft )
    elif left is not None :
      impos[0] = left
    else :
      print( 'define left or dleft')
    if width is not None :
      impos[2] = width
    else :
      print( 'define width' )

  #print( impos )
  return fig.add_axes( impos )

#}}}}}


def create_colorbar_multiaxes( fig, axim0, axim1,
    bottom=None, height=None, left=None, width=None,
         dbottom=None, dleft=None, flag='h') :
  #{{{{{

  # left, bottom, width, height
  impos0 = np.asarray( axim0.get_position().bounds )
  impos = np.asarray( axim0.get_position().bounds )
  impos1 = np.asarray( axim1.get_position().bounds )


  if flag == 'h' :
    impos[3] = height
    if dbottom :
      impos[1] += dbottom
    else :
      impos[1] = bottom 

    impos[2] = impos1[0] + impos1[2] - impos0[0]
  else :
    impos[2] = width
    if dleft :
      impos[0] += dleft
    else :
      impos[0] = left
    impos[3] = impos1[1] + impos1[3] - impos0[1]

  return fig.add_axes( impos )

#}}}}}

def create_axes( fig, widths, heights,  left,  bottom,  dwidth, dheight, dwidths=None, dheights=None, projection=None ) :
  #{{{{{
  nhor = len( widths ) 
  nver = len( heights )

  if dwidths is None :
    dwidths = [ dwidth for _ in range(nhor-1) ]

  if dheights is None :
    dheights =  [ dheight for _ in range(nver-1) ]

  #print( nhor, nver )
  #print( dheights )

  lefts = np.zeros( nhor, dtype=np.float )
  lefts[0]  = left 
  lefts[1:] = np.cumsum(  widths )[:-1] + np.cumsum( dwidths ) + left 

  bottoms = np.zeros( nver, dtype=np.float )
  bottoms[0] = bottom
  bottoms[1:] = ( np.cumsum( np.flipud( heights )  )[:-1] 
                 + np.cumsum( np.flipud( dheights ) ) +  bottom )
  bottoms = np.flipud( bottoms )

  #print( bottoms )


  # create empty 2d array
  axs=[]
  caxs = [ [ None for _ in range( nhor )] for _ in range( nver ) ]
  for iver in range( nver ) :
    #for ihor in range( nhor ) :
    if projection is None :
      axs.append( [ fig.add_axes( [ lefts[ihor], bottoms[iver], 
                                    widths[ihor], heights[iver] ] ) for ihor
                  in range( nhor)  ]  )
    else :
      axs.append( [ fig.add_axes( [ lefts[ihor], bottoms[iver], 
                                    widths[ihor], heights[iver] ],
                                   projection=projection) for ihor
                  in range( nhor)  ]  )
  return  axs, caxs
 

#}}}}}


def remove_axlabels( axs, 
                    xh0=None, xh1=None, xv0=None, xv1=None,
                    yh0=None, yh1=None, yv0=None, yv1=None ) :
  nv = len( axs )
  try :
    nh = len( axs[0] )
  except :
    nh = 1 
  if xh0 is None :
    xh0 =0 
  if xh1 is None :
    xh1 = nh 
  if xv0 is None :
    xv0 =1 
  if xv1 is None :
    xv1 = nv 

  if yh0 is None :
    yh0 = 1 
  if yh1 is None :
    yh1 = nh 
  if yv0 is None :
    yv0 =0 
  if yv1 is None :
    yv1 = nv 
  # remove xs
  for iv in range( xv0, xv1 ) :
    for ih in range( xh0, xh1 ) :
      ax = axs[ iv ][ ih ]
      ax.set_xlabel( '' )
      ax.set_xticklabels( [  ] )

  for iv in range( yv0, yv1 ) :
    for ih in range( yh0, yh1 ) :
      ax = axs[ iv ][ ih ]
      ax.set_ylabel( '' )
      ax.set_yticklabels( [ ] )
def merge_axes( ax0, ax1, flag='v' ) :
  #{{{{{
  # merge two axis in certain directions
  pos0 = ax0.get_position().bounds
  pos1 = ax1.get_position().bounds
  if flag == 'v' :
    height = pos0[1] + pos0[3] - pos1[1]
    pos2 = [ pos0[0], pos1[1], pos0[2], height ]
    ax0.set_position( pos2 )


  ax1.remove()

#}}}}}

class FigFormat : 
#{{{{{
  def __init__( self, figsize=None, widths=None, heights=None, paperwidth=None,
                width_ratios=None, height_ratios=None, left=None, right=None,
                top=None, bottom=None, dwidth=None, dheight=None, 
                aspect=None ) :
    self.figsize = figsize
    self.widths  = widths
    self.heights = heights 
    self.paperwidth = paperwidth 
    self.width_ratios = width_ratios
    self.height_ratios = height_ratios
    self.left = left
    self.right = right 
    self.bottom = bottom
    self.top = top
    self.dwidth = dwidth
    self.dheight = dheight
    self.aspect = aspect
  def calc_figsize_widths_heights( self ) :
    self.figsize, self.widths, self.heights = calc_figsize_widths_heights( 
        self.paperwidth, self.width_ratios, 
        self.height_ratios, self.left, self.right, self.bottom, self.top,
        self.dwidth, self.dheight, self.aspect )
  def create_axes( self, fig ) :
    axs, caxs = create_axes( fig, self.widths, self.heights, self.left, 
        self.bottom, self.dwidth, self.dheight )
    return axs, caxs

  def calc_aspect_from_axis_range( self, axfmt ) :
   self.aspect = abs( axfmt.xmax - axfmt.xmin) / abs( axfmt.ymax - axfmt.ymin )

#}}}}}

class AxesFormat :
  #{{{{{
  def __init__ (self, xlabel=None, ylabel=None, xmin=None, xmax=None,
                ymin=None, ymax=None, 
                xticks=None, yticks=None, cticks=None, 
                xticks_minor=None, yticks_minor=None, cticks_minor=None,
                flag_xminor=0, flag_yminor=0,
                xticklabels=None, yticklabels=None,
                xaxis_loc='top', yaxis_loc='left',
                title=None, subtitle=None, subtitle_position=None,
                subtitle_dir='h', 
                subtitle_halign='center', subtitle_valign='center',
                subtitle_fontsize=12, subtitle_fontweight='bold',
                subtitle_alpha = 0,
                xgrid=True, ygrid=True,
                grid_linewidth=1, grid_linestyle=':',
                fontsize=14,
                vmin=None, vmax=None, cmap='jet'
                ) :
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.xmin   = xmin
    self.xmax   = xmax
    self.ymin   = ymin
    self.ymax   = ymax
    self.xticks  = xticks
    self.yticks  = yticks
    self.cticks  = cticks
    self.flag_xminor = flag_xminor
    self.flag_yminor = flag_yminor
    self.cticks_minor  = cticks_minor
    self.xticks_minor  = xticks_minor
    self.yticks_minor  = yticks_minor
    self.yticklabels = yticklabels
    self.xticklabels = xticklabels

    self.xaxis_loc = xaxis_loc
    self.yaxis_loc = yaxis_loc
    
    self.title  = title
    self.subtitle = subtitle
    self.subtitle_position = subtitle_position
    self.subtitle_dir = subtitle_dir
    self.subtitle_halign= subtitle_halign
    self.subtitle_valign= subtitle_valign
    self.subtitle_alpha = subtitle_alpha
    self.subtitle_fontsize = subtitle_fontsize
    self.subtitle_fontweight = subtitle_fontweight
    
    self.subtitle2 = None
    self.subtitle2_position = [ -0.1, 0.5 ]
    self.subtitle2_dir = 'h'
    self.subtitle2_halign='center'
    self.subtitle2_valign='center'
    self.subtitle2_alpha=0.5
    self.subtitle2_fontweight = 'bold'


  
    self.ygrid = ygrid
    self.xgrid = xgrid
    self.grid_linewidth = grid_linewidth
    self.grid_linestyle = grid_linestyle #':'

    # image only
    self.vmin   = vmin
    self.vmax   = vmax
    self.cmap   = cmap

    # fontsize
    self.fontsize = fontsize


  def format_axes( self, ax ) :


    if self.xlabel : 
      ax.set_xlabel( self.xlabel, fontsize=self.fontsize )

    if self.ylabel :
      ax.set_ylabel( self.ylabel, fontsize=self.fontsize)


    if self.xaxis_loc == 'top' :
      ax.xaxis.tick_top()
      ax.xaxis.set_label_position('top')
 
    if self.yaxis_loc == 'right' :
      ax.yaxis.tick_right()
      ax.yaxis.set_label_position('right')
 
    if type( self.xticks ) is np.ndarray :
      ax.set_xticks( self.xticks )


    if type( self.xticks_minor ) is np.ndarray :
      ax.set_xticks( self.xticks_minor, minor=True )
      self.flag_xminor = 2

    if type( self.yticks ) is np.ndarray :
      ax.set_yticks( self.yticks )

    if type( self.yticks_minor ) is np.ndarray :
      ax.set_yticks( self.yticks_minor, minor=True )
      self.flag_yminor = 2

    if type( self.cticks ) == np.ndarray :
      ax.set_ticks( self.cticks )

    if self.xticklabels :
      ax.set_xticklabels( self.xticklabels )
    elif self.xticklabels == '' :
      ax.set_xticklabels( self.xticklabels )


    if self.yticklabels :
      ax.set_yticklabels( self.yticklabels )
    elif self.yticklabels == '' :
      ax.set_yticklabels( self.yticklabels )

      

    if self.title :
      ax.set_title( self.title, fontsize=self.fontsize+4,
                    fontweight='bold' )

    ax.tick_params( axis='both', which='major', labelsize=self.fontsize )
    if self.flag_yminor == 1 :
      #ax.yaxis.set_minor_locator(AutoMinorLocator())
      ax.tick_params( axis='y', which='both', labelsize=self.fontsize )
      #if ax.yaxis.get_scale() == 'log' :
        #y_minor = LogLocator(base = 10.0,
             #subs = np.arange(1.0, 10.0) * 0.2, numticks = 5 ) 
        #ax.yaxis.set_minor_locator(y_minor)
        #ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if self.flag_xminor == 1 :
      #ax.xaxis.set_minor_locator(AutoMinorLocator())
      ax.tick_params( axis='x', which='both', labelsize=self.fontsize )


    if self.ygrid :
      if self.flag_yminor == 2 :
        ax.yaxis.grid( self.ygrid, linestyle=self.grid_linestyle, which='both', 
            linewidth=self.grid_linewidth)
      else :
        ax.yaxis.grid( self.ygrid, linestyle=self.grid_linestyle, which='major',
            linewidth=self.grid_linewidth)
       

    #print( flag_xminor )
    if self.xgrid :
      if self.flag_xminor == 2 :
        ax.xaxis.grid(self.xgrid, linestyle=self.grid_linestyle, which='both' ,
            linewidth=self.grid_linewidth)
      else :
        ax.xaxis.grid(self.xgrid, linestyle=self.grid_linestyle, which='major' ,
            linewidth=self.grid_linewidth)

    if self.xmin is not None:
      ax.set_xlim( self.xmin, self.xmax )
    if self.ymin is not None:
      ax.set_ylim( self.ymin, self.ymax )

    if self.subtitle :

      if self.subtitle_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      #print( self.subtitle_fontsize )
      if self.subtitle_fontsize is None :
        self.subtitle_fontsize = self.fontsize+2
      
      #print( self.subtitle_fontsize )

      self.ax_subtitle = ax.text( -0.1, 0.5, self.subtitle, 
          fontsize=self.subtitle_fontsize, fontweight=self.subtitle_fontweight,
               bbox=dict(facecolor='w', alpha=self.subtitle_alpha, linewidth=0),
               horizontalalignment=self.subtitle_halign, 
               verticalalignment=self.subtitle_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle.set_position( self.subtitle_position )

    if self.subtitle2 :

      if self.subtitle2_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      self.ax_subtitle2 = ax.text( -0.1, 0.5, self.subtitle2, 
               fontsize=self.fontsize+2, fontweight='bold',
               bbox=dict(facecolor='w', alpha=self.subtitle2_alpha, linewidth=0),
               horizontalalignment=self.subtitle2_halign, 
               verticalalignment=self.subtitle2_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle2.set_position( self.subtitle2_position )



#}}}}}

class CbarAxesFormat :
#{{{{{ 
  def __init__ (self, label=None, xlabel=None, ylabel=None, 
                ticks=None, ticks_minor=None,
                fontsize=14, orientation='horizontal',
                left=None, right=None, bottom=None, dbottom=None,
                width=None,
                height=None, dleft=None) :
    self.label = label
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.ticks = ticks
    self.ticks_minor = ticks_minor
 
    # fontsize
    self.fontsize = fontsize

    # orientation
    self.orientation = orientation

    # axis location
    self.left =left
    self.right=right
    self.bottom=bottom
    self.dbottom=dbottom
    self.dleft=dleft
    self.height=height
    self.width = width

  def create_axes( self, fig, axim ) :
    cax = create_colorbar( fig, axim=axim, caxfmt=self )
    return cax  

  def format_axes( self, cbar ) : # cbar is axis class
    if self.label : 
      if self.orientation == 'horizontal':
          cbar.set_xlabel( self.label, fontsize=self.fontsize, fontweight='regular' )
      if self.orientation == 'vertical':
          cbar.set_ylabel( self.label, fontsize=self.fontsize, fontweight='regular' )
#      cbar.set_label( self.label ) #, fontsize=self.fontsize+2, fontweight='bold' )
#      cbar.set_label( self.label ) #, fontsize=self.fontsize+2, fontweight='bold' )
    if self.xlabel :
      try :
        cbar.set_xlabel( self.xlabel, 
                         fontsize=self.fontsize-2 ) #, fontweight='bold')
      except :
        cbar.ax.set_xlabel( self.xlabel, 
                         fontsize=self.fontsize-2 ) #, fontweight='bold')
 
    if self.ylabel :
      try :
        cbar.set_ylabel( self.ylabel, 
                         fontsize=self.fontsize-2)
      except :
        cbar.ax.set_ylabel( self.ylabel, 
                         fontsize=self.fontsize-2 )
    if type( self.ticks ) is np.ndarray :
      if self.orientation == 'horizontal' :
        cbar.set_xticks( self.ticks)
      if self.orientation == 'vertical' :
        cbar.set_yticks( self.ticks)


    if type( self.ticks_minor ) is np.ndarray :
      if self.orientation == 'vertical' :
        #cbar.yaxis.set_ticks( cbar.norm( self.ticks_minor ), minor=True )
        cbar.yaxis.set_ticks(  self.ticks_minor , minor=True )
      else :
        cbar.xaxis.set_ticks( self.ticks_minor , minor=True )


    cbar.tick_params( axis='both', which='major', labelsize=self.fontsize )
    cbar.tick_params( labelsize=self.fontsize )

#}}}}}

class GeoAxesFormat : 
#{{{{{
  def __init__ (self) :
    self.xlabel = None
    self.ylabel = None
    self.xmin   = None 
    self.xmax   = None
    self.ymin   = None
    self.ymax   = None
    self.xticks  = None
    self.yticks  = None 
    self.cticks  = None 
    self.cticks_minor  = None 
    self.xticks_minor  = None
    self.yticks_minor  = None 
    self.yticklabels = None
    self.xticklabels = None

    self.xaxis_loc = None
    
    self.title  = None
    self.subtitle = None
    self.subtitle_position = [ -0.1, 0.5 ]
    self.subtitle_dir = 'v'
    self.subtitle_halign='center'
    self.subtitle_valign='center'
    self.subtitle_alpha=0.5
    
    self.subtitle2 = None
    self.subtitle2_position = [ -0.1, 0.5 ]
    self.subtitle2_dir = 'h'
    self.subtitle2_halign='center'
    self.subtitle2_valign='center'
    self.subtitle2_alpha=0.5


  
    self.ygrid = False
    self.xgrid = False

    # image only
    self.vmin   = None
    self.vmax   = None

    # fontsize
    self.fontsize = 18


  def format_axes( self, ax ) :
    flag_yminor = 0
    flag_xminor = 0


    if self.xlabel : 
      #ax.set_xlabel( self.xlabel, fontsize=self.fontsize )
      if self.xaxis_loc == 'top' :
        ax.text( 0.5, 1.1, self.xlabel,  
                fontsize=self.fontsize, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center')
      else :
        ax.text( 0.5, -0.1, self.xlabel,  
                fontsize=self.fontsize, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center')

    if self.ylabel :
      #ax.set_ylabel( self.ylabel, fontsize=self.fontsize)
      ax.text( -0.1, 0.5, self.ylabel,  
              fontsize=self.fontsize, transform=ax.transAxes,
              rotation='vertical',
              horizontalalignment='center', verticalalignment='center')

 


    if type( self.xticks_minor ) is np.ndarray :
      ax.set_xticks( self.xticks_minor, minor=True )
      flag_xminor = 1


    if type( self.yticks_minor ) is np.ndarray :
      ax.set_yticks( self.yticks_minor, minor=True )
      flag_yminor = 1


    if self.xticklabels :
      ax.set_xticklabels( self.xticklabels )
    elif self.xticklabels == '' :
      ax.set_xticklabels( self.xticklabels )


    if self.yticklabels :
      ax.set_yticklabels( self.yticklabels )
    elif self.yticklabels == '' :
      ax.set_yticklabels( self.yticklabels )

      

    if self.title :
      ax.set_title( self.title, fontsize=self.fontsize+4,
                    fontweight='bold' )

    if type( self.yticks ) is np.ndarray :
      ylocator = mticker.FixedLocator( self.yticks )
    else :
      ylocator = None
    ##    
    if type( self.xticks ) is np.ndarray :
      xlocator = mticker.FixedLocator( self.xticks )
    else :
      xlocator = None

    if self.xgrid or self.ygrid :
      gl=ax.gridlines( linestyle=':', draw_labels=True,
                       xlocs=xlocator, ylocs=ylocator )


 
    if self.xgrid :
      gl.xlines = True
    else :
      gl.xlines = False

    if self.ygrid :
      gl.ylines = True
    else :
      gl.ylines = False
#
    gl.xlabel_style = { 'size': self.fontsize }
    gl.ylabel_style = { 'size': self.fontsize }
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if self.xaxis_loc == 'top' :
      gl.xlabels_bottom = False
    else :
      gl.xlabels_top = False

    gl.ylabels_right = False
#
    #print ( gl.xline_artists )
    #print ( gl.yline_artists )


    #ax.set_extent( [ self.xmin, self.xmax, self.ymin, self.ymax ],
    #                crs=self.crs)
    
    ax.set_extent( [ self.xmin, self.xmax, self.ymin, self.ymax ] )
                    #crs=self.crs)
                    #crs=ccrs.Geodetic())
    #if self.ygrid :
    #  if flag_yminor == 1 :
    #    ax.yaxis.grid( self.ygrid, linestyle=':', which='both' )
    #  else :
    #    ax.yaxis.grid( self.ygrid, linestyle=':', which='major' )
       

    #print( flag_xminor )
      #if flag_xminor == 1 :
      #  ax.xaxis.grid(self.xgrid, linestyle=':', which='both' )
      #else :
      #  ax.xaxis.grid(self.xgrid, linestyle=':', which='major' )

    #ax.set_xlim( self.xmin, self.xmax )
    #ax.set_ylim( self.ymin, self.ymax )

    if self.subtitle :

      if self.subtitle_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      self.ax_subtitle = ax.text( -0.1, 0.5, self.subtitle, 
               fontsize=self.fontsize+2, fontweight='bold',
               bbox=dict(facecolor='w', alpha=self.subtitle_alpha, linewidth=0),
               horizontalalignment=self.subtitle_halign, 
               verticalalignment=self.subtitle_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle.set_position( self.subtitle_position )

    if self.subtitle2 :

      if self.subtitle2_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      self.ax_subtitle2 = ax.text( -0.1, 0.5, self.subtitle2, 
               fontsize=self.fontsize+2, fontweight='bold',
               bbox=dict(facecolor='w', alpha=self.subtitle2_alpha, linewidth=0),
               horizontalalignment=self.subtitle2_halign, 
               verticalalignment=self.subtitle2_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle2.set_position( self.subtitle2_position )


def remove_xlabels_subplot( axs, nv, nh) :
  for iv in range( 1, nv ) :
    for ih in range( 0, nh ) :
      ax = axs[iv][ih]
      if ax.figure is not None : 
        ax.set_xlabel('')
        ax.set_xticklabels([] )

def remove_ylabels_subplot( axs, nv, nh) :
  for iv in range( 0, nv ) :
    for ih in range( 1, nh ) :
      ax = axs[iv][ih]
      if ax.figure is not None : 
        ax.set_ylabel('')
        ax.set_yticklabels([] )
