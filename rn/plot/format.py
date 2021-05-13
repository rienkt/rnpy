import numpy as np 
from matplotlib import rcParams
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
  nfig_hor = len( width_ratios )
  nfig_ver = len( height_ratios )

  width_ratios = np.asarray( width_ratios, dtype=np.float )
  height_ratios = np.asarray( height_ratios, dtype=np.float )
  if dwidths is None :
    dwidths = np.ones( nfig_hor -1 , dtype=np.float ) * dwidth
  else :
    dwidths = np.asarray( dwidths, dtype=np.float )
  print( nfig_hor, dwidths )

  widths = ( ( 1. - left - right - np.sum(dwidths) ) *
             width_ratios / np.sum( width_ratios ) )
  if dheights is None :
    dheights = np.ones( nfig_ver - 1, dtype=np.float ) * dheight
    print( nfig_ver, dheights )
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
def create_colorbar( fig, axim=None, 
    bottom=None, height=None, left=None, width=None,
    flag='h' ) :
  #{{{{{

  if axim :
    impos = np.asarray( axim.get_position().bounds )
    axheight = impos[3]
    axwidth  = impos[2]
    if bottom :
      bottom *= axheight
    if height :
      height *= axheight
    if left :
      left *= axwidth
    if width :
      width *= axwidth
  else :
    impos = np.asarray( [ 0., 0., 1., 1. ] )


  if flag == 'h' :
    impos[3] = height 
    if bottom :
      impos[1] += bottom 
    if width :
      impos[2] = width 
    if left :
      impos[0] += left
  else :
    if left :
      impos[0] += ( impos[2] + left )
    impos[2] = width

  print( impos )
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

  print( nhor, nver )
  print( dheights )

  lefts = np.zeros( nhor, dtype=np.float )
  lefts[0]  = left 
  lefts[1:] = np.cumsum(  widths )[:-1] + np.cumsum( dwidths ) + left 

  bottoms = np.zeros( nver, dtype=np.float )
  bottoms[0] = bottom
  bottoms[1:] = ( np.cumsum( np.flipud( heights )  )[:-1] 
                 + np.cumsum( np.flipud( dheights ) ) +  bottom )
  bottoms = np.flipud( bottoms )

  print( bottoms )


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


class AxesFormat :
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
    self.yaxis_loc = None
    
    self.title  = None
    self.subtitle = None
    self.subtitle_position = [ -0.1, 0.5 ]
    self.subtitle_dir = 'v'
    self.subtitle_halign='center'
    self.subtitle_valign='center'
    self.subtitle_alpha=0.5
    self.subtitle_fontsize = None
    self.subtitle_fontweight = 'bold'
    
    self.subtitle2 = None
    self.subtitle2_position = [ -0.1, 0.5 ]
    self.subtitle2_dir = 'h'
    self.subtitle2_halign='center'
    self.subtitle2_valign='center'
    self.subtitle2_alpha=0.5
    self.subtitle2_fontweight = 'bold'


  
    self.ygrid = False
    self.xgrid = False
    self.grid_linewidth = 1
    self.grid_linestyle = ':'

    # image only
    self.vmin   = None
    self.vmax   = None

    # fontsize
    self.fontsize = 18


  def format_axes( self, ax ) :
    flag_yminor = 0
    flag_xminor = 0


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
      flag_xminor = 1

    if type( self.yticks ) is np.ndarray :
      ax.set_yticks( self.yticks )

    if type( self.yticks_minor ) is np.ndarray :
      ax.set_yticks( self.yticks_minor, minor=True )
      flag_yminor = 1

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

    if self.ygrid :
      if flag_yminor == 1 :
        ax.yaxis.grid( self.ygrid, linestyle=self.grid_linestyle, which='both', 
            linewidth=self.grid_linewidth)
      else :
        ax.yaxis.grid( self.ygrid, linestyle=self.grid_linestyle, which='major',
            linewidth=self.grid_linewidth)
       

    print( flag_xminor )
    if self.xgrid :
      if flag_xminor == 1 :
        ax.xaxis.grid(self.xgrid, linestyle=self.grid_linestyle, which='both' ,
            linewidth=self.grid_linewidth)
      else :
        ax.xaxis.grid(self.xgrid, linestyle=self.grid_linestyle, which='major' ,
            linewidth=self.grid_linewidth)

    print( self.xmin, self.ymin, 'tehe' )
    if self.xmin is not None:
      print('a m')
      ax.set_xlim( self.xmin, self.xmax )
    if self.ymin is not None:
      ax.set_ylim( self.ymin, self.ymax )

    if self.subtitle :

      if self.subtitle_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      print( self.subtitle_fontsize )
      if self.subtitle_fontsize is None :
        self.subtitle_fontsize = self.fontsize+2
      
      print( self.subtitle_fontsize )

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
  def __init__ (self) :
    self.label = None
    self.xlabel = None
    self.ylabel = None
    self.ticks = None
    self.ticks_minor = None   
 
    # fontsize
    self.fontsize = 14

    # orientation
    self.orientation = 'horizontal'


  def format_axes( self, cbar ) : # cbar is colorbar class
    if self.label : 
      try :
        cbar.set_label( self.label, fontsize=self.fontsize+2, fontweight='bold' )
      except :
        cbar.set_label( self.label, size=self.fontsize+2, fontweight='bold' )
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
        cbar.set_ticks( self.ticks)
      if self.orientation == 'vertical' :
        cbar.set_ticks( self.ticks)


    if type( self.ticks_minor ) is np.ndarray :
      if cbar.orientation == 'vertical' :
        cbar.ax.yaxis.set_ticks( cbar.norm( self.ticks_minor ), minor=True )
      else :
        cbar.ax.xaxis.set_ticks( cbar.norm( self.ticks_minor ), minor=True )


    cbar.ax.tick_params( axis='both', which='major', labelsize=self.fontsize )
    cbar.ax.tick_params( labelsize=self.fontsize )

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
    print ( gl.xline_artists )
    print ( gl.yline_artists )


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


