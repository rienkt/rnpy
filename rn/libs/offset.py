#!/usr/bin/env python
from __future__ import print_function
"""

  <MODULE NAME>

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Kamei

"""
__author__ = "Rie Kamei"

#======================================================================
# Modules
#======================================================================

import numpy as np



#======================================================================
# classes / functions
#======================================================================


def offset_weighting( data, offset, min_offset, max_offset ) :
  # this implements a hard-limit offset weighting 
  # if flag=1, we include data at min_offset < offset < max_offset
 
  data[ offset < min_offset ] = 0.
  data[ offset > max_offset ] = 0.
  return data 
