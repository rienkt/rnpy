#!/usr/bin/env python
from __future__ import print_function
"""


  <MODULE NAME>

  <ADD_DESCRIPTION_HERE>


  * PARAMETERS




  Author : Rie Nakata (Kamei)

"""
__author__ = "Rie Nakata (Kamei)"

#==============================================================================
#                                                                    MODULES 
#==============================================================================

#import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
import sys
#import copy

#import rsf.api as rsf

#import rk.rsf.model as rk_model
#import rk.bin.active.bin as rk_bin
#import rk.bin.active.process as rk_process



try :
  from mpi4py import MPI

  comm = MPI.COMM_WORLD
  nproc = comm.Get_size()
  rank = comm.Get_rank()

except :
  print( 'warning: mpi4py does not exist' )

#==============================================================================
#                                                        CLASSES / FUNCTIONS
#==============================================================================



def eprint( *args ) :
  print( args, file=sys.stderr )


def wprint( *args ) :
  print( 'WARNING : ', args )

def mpi_print(*args, sep=' ', end='\n', file=None, flush=True):
  #if rank == 63 :
    prefix = f'rank {rank}/{nproc}: '
    print(prefix + sep.join(str(arg) for arg in args),
              end=end, file=file, flush=flush)

