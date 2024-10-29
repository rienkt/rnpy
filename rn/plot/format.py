import numpy as np 
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator)
import matplotlib
from matplotlib import ticker

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
#rcParams['font.sans-serif'] = 'Helvetica'
rcParams['pdf.fonttype'] = 42

import matplotlib.ticker as mticker

try :
  from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
  import cartopy.crs as ccrs

except :
  print( 'cartopy is not available' )
  #rcParams['png.fonttype'] = 42

subtitles = [ '(a)', '(b)', '(c)', '(d)', '(e)',
              '(f)', '(g)', '(h)', '(i)',  '(j)',
              '(k)', '(l)', '(m)','(n)', '(o)', '(p)','(q)','(r)' ]



def add_text( ax, x, y, text, fontsize=12, fontweight='regular', 
              color='k', bbox_alpha=0.5, halign='center',
              bbox_linewidth=0, bbox_style='square',
              valign='center' , rotation=0) :
      position = [ x, y ]
      #print( fontweight )
      ax_subtitle = ax.text( -0.1, 0.5, text, 
               fontsize=fontsize, 
               weight=fontweight,
               color = color,
               bbox=dict( facecolor='w', 
                          alpha=bbox_alpha, linewidth=bbox_linewidth,
                          boxstyle=bbox_style),
               horizontalalignment=halign, 
               verticalalignment=valign,
               transform=ax.transAxes, 
               rotation=rotation )
      ax_subtitle.set_position( position )
      return ax_subtitle


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

  width_ratios = np.asarray( width_ratios, dtype=float )
  height_ratios = np.asarray( height_ratios, dtype=float )
  if dwidths is None :
    dwidths = np.ones( nfig_hor -1 , dtype=float ) * dwidth
  else :
    dwidths = np.asarray( dwidths, dtype=float )
  #print( nfig_hor, dwidths )

  widths = ( ( 1. - left - right - np.sum(dwidths) ) *
             width_ratios / np.sum( width_ratios ) )
  if dheights is None :
    dheights = np.ones( nfig_ver - 1, dtype=float ) * dheight
    #print( nfig_ver, dheights )
  else :
    dheights = np.asarray( dheights, dtype=float )
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
    impos[3] = height 
    if dbottom :
      impos[1] += dbottom
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
         dbottom=None, dleft=None, dright=None, flag='h') :
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
    if dright :
      impos[2] += dright
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

  lefts = np.zeros( nhor, dtype=float )
  lefts[0]  = left 
  lefts[1:] = np.cumsum(  widths )[:-1] + np.cumsum( dwidths ) + left 

  bottoms = np.zeros( nver, dtype=float )
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
                    yh0=None, yh1=None, yv0=None, yv1=None,
                    xaxis = True, yaxis=True, axfmt=None, xlabel_loc='top',
                    ylabel_loc = 'left',
                    xticklabels = True, yticklabels=True) :
  print( axfmt )
  nv = len( axs )
  try :
    nh = len( axs[0] )
  except :
    nh = 1
  if axfmt : 
    xlabel_loc = axfmt.xaxis_loc
    ylabel_loc = axfmt.yaxis_loc

  if xh0 is None :
    xh0 =0 
  if xh1 is None :
    xh1 = nh  -1
  if xlabel_loc == 'top' :
    if xv0 is None :
      xv0 =1 
    if xv1 is None :
      xv1 = nv  -1
  else :
    if xv0 is None :
      xv0 = 0
    if xv1 is None :
      xv1 = nv -1

  if yh0 is None :
    yh0 = 1 
  if yh1 is None :
    yh1 = nh -1
  if yv0 is None :
    yv0 =0 
  if yv1 is None :
    yv1 = nv -1 
  # remove xs
  if xaxis  :
    for iv in range( xv0, xv1 +1 ) :
      for ih in range( xh0, xh1 +1) :
        try :
          ax = axs[ iv ][ ih ]
          ax.set_xlabel( '' )
        except :
          print('ax: %d %d does not exist '%( iv, ih ))
  if xticklabels :
    for iv in range( xv0, xv1 +1 ) :
      for ih in range( xh0, xh1 +1) :
        try :
          ax = axs[ iv ][ ih ]
          ax.set_xticklabels( [  ] )
        except :
          print('ax: %d %d does not exist '%( iv, ih ))
  if yaxis :
    for iv in range( yv0, yv1  + 1) :
      for ih in range( yh0, yh1 + 1) :
        try :
          ax = axs[ iv ][ ih ]
          ax.set_ylabel( '' )
        except :
          print('ax: %d %d does not exist '%( iv, ih ))
  if yticklabels:
    for iv in range( yv0, yv1  + 1) :
      for ih in range( yh0, yh1 + 1) :
        try :
          ax = axs[ iv ][ ih ]
          ax.set_yticklabels( [ ] )
        except :
          print('ax: %d %d does not exist '%( iv, ih ))
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
                top=None, bottom=None, dwidths=None, dwidth=None, dheight=None,
                dheights=None, aspect=None, projection=None ) :
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
    self.dwidths = dwidths
    self.dheight = dheight
    self.dheights = dheights
    self.aspect = aspect
    self.projection = projection
    if self.widths is None :
      self.calc_figsize_widths_heights()
  def calc_figsize_widths_heights( self ) :
    self.figsize, self.widths, self.heights = calc_figsize_widths_heights( 
        self.paperwidth, self.width_ratios, 
        self.height_ratios, self.left, self.right, self.bottom, self.top,
        self.dwidth, self.dheight, self.aspect , dwidths=self.dwidths, 
        dheights=self.dheights)
  def create_axes( self, fig, projection=None ) :
    if self.widths is None :
      self.calc_figsize_widths_heights()
    axs, caxs = create_axes( fig, self.widths, self.heights, self.left, 
        self.bottom, self.dwidth, self.dheight, dwidths=self.dwidths, 
        dheights=self.dheights, projection=projection )
    return axs, caxs

  def calc_aspect_from_axis_range( self, axfmt ) :
   self.aspect = abs( axfmt.xmax - axfmt.xmin) / abs( axfmt.ymax - axfmt.ymin )

#}}}}}

class AxesFormat :
  #{{{{{
  def __init__ (self, 
                xlabel=None, ylabel=None, rlabel=None,

                xlabel_pad = None, ylabel_pad = None,
                xmin=None, xmax=None,
                ymin=None, ymax=None, 
                rmin=None, rmax=None,
                xticks=None, yticks=None, cticks=None, rticks=None,
                xscale=None, yscale=None, rscale=None,
                xticks_minor=None, yticks_minor=None, cticks_minor=None,
                rticks_minor=None,
                flag_xminor=0, flag_yminor=0,flag_rminor=0,
                xticklabels=None, yticklabels=None,rticklabels=None,
                xaxis_loc=None, yaxis_loc=None, raxis_loc='None',
                xticklabels_format = None, yticklabels_format=None,
                title=None, 
                subtitle=None, subtitle_position=None,
                subtitle_dir='h', 
                subtitle_halign='center', subtitle_valign='center',
                subtitle_fontsize=12, subtitle_fontweight='bold',
                subtitle_color = 'k', 
                subtitle_bbox_alpha = 0,
                subtitle2=None, subtitle2_position=[0.5,0.5],
                subtitle2_dir='h', 
                subtitle2_halign='center', subtitle2_valign='center',
                subtitle2_fontsize=12, subtitle2_fontweight='bold',
                subtitle2_color = 'k', 
                subtitle2_bbox_alpha = 0,
                subtitle3=None, subtitle3_position=None,
                subtitle3_dir='h', 
                subtitle3_halign='center', subtitle3_valign='center',
                subtitle3_fontsize=12, subtitle3_fontweight='bold',
                subtitle3_color = 'k', 
                subtitle3_bbox_alpha = 0,
                xgrid=True, ygrid=True,rgrid=True,
                grid_linewidth=1, grid_linestyle=':',
                fontsize=14, fontweight='regular',
                vmin=None, vmax=None, cmap='jet',
                polar=False,
                ) :
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.xlabel_pad = xlabel_pad
    self.ylabel_pad = ylabel_pad

    self.xmin   = xmin
    self.xmax   = xmax
    self.ymin   = ymin
    self.ymax   = ymax
    self.rmin   = rmin
    self.rmax   = rmax
    self.xscale = xscale
    self.yscale = yscale
    self.rscale = rscale
    self.xticks  = xticks
    self.yticks  = yticks
    self.rticks  = rticks
    self.cticks  = cticks
    self.flag_xminor = flag_xminor
    self.flag_yminor = flag_yminor
    self.flag_rminor = flag_rminor
    self.cticks_minor  = cticks_minor
    self.xticks_minor  = xticks_minor
    self.yticks_minor  = yticks_minor
    self.rticks_minor  = rticks_minor
    self.yticklabels = yticklabels
    self.rticklabels = rticklabels
    self.xticklabels = xticklabels
    self.xticklabels_format = xticklabels_format
    self.yticklabels_format = yticklabels_format

    self.xaxis_loc = xaxis_loc
    self.yaxis_loc = yaxis_loc
    self.raxis_loc = raxis_loc


    self.polar = polar
    
    self.title  = title
    self.subtitle2 = subtitle2
    self.subtitle2_position = subtitle2_position
    self.subtitle2_dir = subtitle2_dir
    self.subtitle2_halign= subtitle2_halign
    self.subtitle2_valign= subtitle2_valign
    self.subtitle2_bbox_alpha = subtitle2_bbox_alpha
    self.subtitle2_fontsize = subtitle2_fontsize
    self.subtitle2_fontweight = subtitle2_fontweight
    self.subtitle2_color = subtitle2_color
    
    self.subtitle = subtitle
    self.subtitle_position = subtitle_position
    self.subtitle_dir = subtitle_dir
    self.subtitle_halign= subtitle_halign
    self.subtitle_valign= subtitle_valign
    self.subtitle_bbox_alpha = subtitle_bbox_alpha
    self.subtitle_fontsize = subtitle_fontsize
    self.subtitle_fontweight = subtitle_fontweight
    self.subtitle_color = subtitle_color

    self.subtitle3 = subtitle3
    self.subtitle3_position = subtitle3_position
    self.subtitle3_dir = subtitle3_dir
    self.subtitle3_halign= subtitle3_halign
    self.subtitle3_valign= subtitle3_valign
    self.subtitle3_bbox_alpha = subtitle3_bbox_alpha
    self.subtitle3_fontsize = subtitle3_fontsize
    self.subtitle3_fontweight = subtitle3_fontweight
    self.subtitle3_color = subtitle3_color
  
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
    self.fontweight = fontweight


  def format_axes( self, ax ) :

    ax.tick_params( 
      bottom=True, top=True, 
      left=True, right=True )

    if self.polar == False :
      if self.xlabel : 
        ax.set_xlabel( self.xlabel, fontsize=self.fontsize, fontweight=self.fontweight )




      if self.ylabel :
        ax.set_ylabel( self.ylabel, fontsize=self.fontsize, fontweight=self.fontweight)


      if self.xaxis_loc == 'top' :
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
   
      if self.yaxis_loc == 'right' :
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

      if self.xscale :
        ax.set_xscale( self.xscale )
      if self.yscale :
        ax.set_yscale( self.yscale )
 
      if type( self.cticks ) == np.ndarray :
        ax.set_ticks( self.cticks )
        #self.flag_xminor = 2


      if self.xticklabels :
        ax.set_xticklabels( self.xticklabels )
      elif self.xticklabels == '' :
        ax.set_xticklabels( self.xticklabels )

      
      ax.tick_params( axis='both', which='major', labelsize=self.fontsize )

    if type( self.xticks ) is np.ndarray :
      ax.set_xticks( self.xticks )
    if type( self.xticks_minor ) is np.ndarray :
      ax.set_xticks( self.xticks_minor, minor=True )
    if type( self.yticks ) is np.ndarray :
      ax.set_yticks( self.yticks )

    if type( self.yticks_minor ) is np.ndarray :
      ax.set_yticks( self.yticks_minor, minor=True )
      #self.flag_yminor = 2
    if self.yticklabels :
      ax.set_yticklabels( self.yticklabels )
    elif self.yticklabels == '' :
      ax.set_yticklabels( self.yticklabels )



    if self.polar == False :
      if self.flag_yminor == 1 :
        ax.tick_params( axis='y', which='both', labelsize=self.fontsize )
      if self.flag_xminor == 1 :
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

    if self.title :
      ax.set_title( self.title, fontsize=self.fontsize+4,
                    fontweight='bold' )

    if self.polar == 1 :
      self.xgrid = None
      self.ygrid = None


    if self.polar is False :
      if self.xmin is not None:
        ax.set_xlim( self.xmin, self.xmax )
      if self.ymin is not None:
        ax.set_ylim( self.ymin, self.ymax )

    if self.rmin is not None:
      #print( self.rmin, self.rmax )
      ax.set_rlim( self.rmin, self.rmax )

    if self.fontweight is not None :
      #print( self.fontweight )
      ax.set_yticklabels(ax.get_yticks(), weight=self.fontweight)
      ax.set_xticklabels(ax.get_xticks(), weight=self.fontweight)
      ax.set_xlabel( ax.get_xlabel(), weight=self.fontweight )
      ax.set_ylabel( ax.get_ylabel(), weight=self.fontweight )

    if self.xlabel_pad is not None :
      ax.set_xlabel( ax.get_xlabel(), rotation=0, labelpad=self.xlabel_pad )
    if self.ylabel_pad is not None :
      ax.set_ylabel( ax.get_ylabel(), rotation=90, labelpad=self.ylabel_pad )

    # tick labels format
    if self.xticklabels_format is not None :
      ax.xaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:%s}"%(self.xticklabels_format)) )

    if self.yticklabels_format is not None :
      ax.yaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:%s}"%(self.yticklabels_format)) )

    if self.subtitle :

      if self.subtitle_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      #print( self.subtitle_fontsize )
      if self.subtitle_fontsize is None :
        self.subtitle_fontsize = self.fontsize
      
      #print( self.subtitle_fontsize )

      self.ax_subtitle = ax.text( -0.1, 0.5, self.subtitle, 
               fontsize=self.subtitle_fontsize, 
               fontweight=self.subtitle_fontweight,
               color = self.subtitle_color,
               bbox=dict( facecolor='w', 
                          alpha=self.subtitle_bbox_alpha, linewidth=0),
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
               fontsize=self.subtitle2_fontsize, 
               fontweight=self.subtitle2_fontweight,
               bbox=dict(facecolor='w', alpha=self.subtitle2_bbox_alpha, linewidth=0),
               horizontalalignment=self.subtitle2_halign, 
               verticalalignment=self.subtitle2_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle2.set_position( self.subtitle2_position )

    if self.subtitle3 :

      if self.subtitle3_dir == 'v'  :
          rotation = 90
      else :
          rotation = 0

      self.ax_subtitle3 = ax.text( -0.1, 0.5, self.subtitle3, 
               fontsize=self.subtitle3_fontsize, 
               fontweight=self.subtitle3_fontweight,
               bbox=dict(facecolor='w', alpha=self.subtitle3_bbox_alpha, linewidth=0),
               horizontalalignment=self.subtitle3_halign, 
               verticalalignment=self.subtitle3_valign,
               transform=ax.transAxes, 
               rotation=rotation )
      self.ax_subtitle3.set_position( self.subtitle3_position )


#}}}}}

class CbarAxesFormat :
#{{{{{ 
  def __init__ (self, label=None, xlabel=None, ylabel=None, 
                xaxis_loc ='top', 
                ticks=None, ticks_minor=None,
                fontsize=14, orientation='horizontal',
                left=None, right=None, bottom=None, dbottom=None,
                width=None,
                height=None, dleft=None) :
    self.label = label
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.xaxis_loc = xaxis_loc
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
    if self.xlabel :
      try :
        cbar.set_xlabel( self.xlabel, 
                         fontsize=self.fontsize ) #, fontweight='bold')
      except :
        cbar.ax.set_xlabel( self.xlabel, 
                         fontsize=self.fontsize ) #, fontweight='bold')
 
    if self.ylabel :
      try :
        cbar.set_ylabel( self.ylabel, 
                         fontsize=self.fontsize)
      except :
        cbar.ax.set_ylabel( self.ylabel, 
                         fontsize=self.fontsize )


    if self.xaxis_loc == 'top' :
        cbar.xaxis.tick_top()
        cbar.xaxis.set_label_position('top')
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
