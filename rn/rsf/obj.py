#!/usr/bin/env python

# Import RSF structure (like SEP.top)
import rsf.api as rsf

# Import math packages
import numpy as np

# define function
class Obj:
  def __init__( self ) :
    self.niter = 1 
    self.diter = 0.
    self.oiter = 0.
    self.nblock = 1
    self.dblock = 1.
    self.oblock = 1.


  def read( self, input=None, f=None):
    if input is None :
      input = rsf.Input(f)
    self.niter = input.int('n1')
    self.diter = input.float('d1')
    self.oiter = input.float('o1')
    if (input.int('n2')):
      self.nblock = input.int('n2')
      self.dblock = input.float('d2')
      self.oblock = input.float('o2')

    self.data = np.zeros((self.nblock, self.niter), 'f')

    input.read(self.data)
    self.set_axis()

  def write(self,output):
    output.put('n1', self.niter)
    output.put('n2', self.nblock)
    output.put('d1', self.diter)
    output.put('d2', self.dblock)
    output.put('o1', self.oiter)
    output.put('o2', self.oblock)
    output.write(self.data)
    output.close()
  def set_axis(self):
    self.block = np.arange(0, self.nblock, dtype=np.float) * self.dblock + self.oblock
    self.iter  = np.arange(0,self.niter,dtype=np.float) * self.diter + self.oiter

