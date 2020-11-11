import numpy as np

def mad( d, axis=None ) :
  if axis is None :
    return np.median( np.abs( d - np.median(d) ) )
  else : 
    if axis != -1 and axis != ( d.ndim - 1 ) :
      d = np.swapaxes( d, axis, -1 )
      nd = d.shape[-1]
      d_med = np.repeat( np.expand_dims( np.median( d, axis=-1 ), axis=-1 ), 
                                                          nd, axis=-1 )
      return np.swapaxes( np.median( np.abs( d - d_med ) , axis=-1 ), axis, -1 )
    else :
      nd = d.shape[-1]
      d_med = np.repeat( np.expand_dims( np.median( d, axis=-1 ), axis=-1 ), 
                                                          nd, axis=-1 )
      return  np.median( np.abs( d - d_med ) , axis=-1 )

