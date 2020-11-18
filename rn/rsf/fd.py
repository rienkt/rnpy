#!/usr/bin/env python

# Import RSF structure (like SEP.top)
import rsf.api as rsf
import os.path
import os
import numpy as np
import rn.bin.active.core_nocsv as rn_bin
      
def get_dim(input):
    dim=0
    for i in range(1,10):
        if input.int('n'+str(i))==None:
            break
        else:
            dim=i
    return dim

class rn_loc( rn_bin.rn_loc ) :
  def __init__( self, n=1, ref=None) :
    rn_bin.rn_loc.__init__( self, n=n, ref=ref )
    self.d = 1.
    self.o = 0.
  def read( self, floc=None ) :
    if floc :
      self.fname = floc

    if os.path.isfile( self.fname ):
      try :
        self.x = np.fromfile( floc, sep = ' ', count = self.n * 2
                    ).reshape( [self.n, 2] )[:, 0]
        self.z = np.fromfile( floc, sep = ' ', count = self.n * 2 
                    ).reshape( [self.n, 2] )[:, 1]
      except :
        self.x = np.arange( 0, self.n, dtype='f' )*self.d + self.o
        self.z = np.arange( 0, self.n, dtype='f' )*self.d + self.o
    else:
      print( '%s does not exist'%self.fname ) 
      self.x = np.arange( 0, self.n, dtype='f' )*self.d + self.o
      self.z = np.arange( 0, self.n, dtype='f' )*self.d + self.o
  def m2km( self ):
    self.x *= 1e-3
    self.z *= 1e-3

    #if self.scale is not None : 
    #  self.x = self.x * self.scale
    #  self.z = self.z * self.scale

class rn_freq( rn_bin.rn_freq ) :
  def __init__( self, n=1 ) :
    rn_bin.rn_freq.__init__( self, n )
  def read( self, f=None ) :
    if f :
      self.fname = f

    with open( os.path.join( self.fdir, self.fname ) ) as f :
      lines = f.read().splitlines()
   
    n = int( lines[0]  )
    self.set_n( n )
    
    self.d = np.array( [ line.split()[0] for line in lines[1:n+1] ],
                        dtype=np.float )
    

# define function     
class FreqdomainData:
  def __init__(self,ref=None, nsrc=1, nrcv=1, niter=1, nfreq=1 ) :
    self.rcvs = rn_loc()
    self.srcs = rn_loc()
    self.freqs = rn_freq()
    if ref != None:
      self.srcs = ref.srcs
      self.rcvs = ref.rcvs
      self.freqs = ref.freqs 
      self.niter = ref.niter
      self.scale = ref.scale
    else:
      self.rcvs.n = nrcv
      self.srcs.n = nsrc
      self.freqs.n = nfreq
      self.niter = niter
      self.scale = 1.0
  def initialize(self, gather = 'all', init_value = 0.):
    if gather == 's' :
      self.d = np.ones((self.freqs.n, self.rcvs.n), 'c16') * init_value
    elif gather=='r':
      self.d = np.ones((self.freqs.n, self.srcs.n ), 'c16') * init_value
    elif gather=='f':
      self.d = np.ones((self.srcs.n,  self.rcvs.n), 'c16') * init_value
    else:
      self.d = np.ones( (self.freqs.n, self.srcs.n, self.rcvs.n), 'c16'
                      ) * init_value

  def initialise(self, gather = 'all', init_value = 0.):
    self.initialize(gather,init_value)

  def read_header(self, fre, fim,
                fsrc='src.txt', frcv='rcv.txt',  ffreq='freq.txt')  :
    # sfile='src.dat', rfile='rcv.dat', ffile='freq.dat'):

    input_re = rsf.Input(fre)
    input_im = rsf.Input(fim)
    self.rcvs.n    = input_re.int('n1')
    self.srcs.n    = input_re.int('n2')
    self.rcvs.d    = input_re.float('d1')
    self.srcs.d    = input_re.float('d2')
    self.rcvs.o    = input_re.float('o1')
    self.srcs.o    = input_re.float('o2')

    if (get_dim(input_re) == 3):
      self.freqs.n = input_re.int('n3')
      self.freqs.d = input_re.float('d3')
      self.freqs.o = input_re.float('o3')
    else:
      self.freqs.n = 1
      self.freqs.d = 1.
      self.freqs.o = 1.

   
    if os.path.exists( fsrc ) :
      self.srcs.read( fsrc )
    if os.path.exists( frcv ) :
      self.rcvs.read( frcv )
    if os.path.exists( ffreq ) :
      self.freqs.read( ffreq )


  def read( self, fre, fim, gather='all',
            fsrc='src.txt', frcv='rcv.txt',  ffreq='freq.txt',
            inum=1 ) :
            #sfile='src.dat', rfile='rcv.dat', ffile='freq.dat'):
    input_re=rsf.Input(fre)
    input_im=rsf.Input(fim)
    self.read_header(fre,fim,fsrc,frcv,ffreq)

    tmp_re=np.zeros((self.srcs.n,self.rcvs.n),'f')
    tmp_im=np.zeros((self.srcs.n,self.rcvs.n),'f')
    if gather=='s' :
      isrc=inum
      tmpd_real=np.zeros((self.freqs.n,self.rcvs.n),'f')
      tmpd_imag=np.zeros((self.freqs.n,self.rcvs.n),'f')
      for ifreq in range(self.freqs.n):
        input_re.read(tmp_re)
        input_im.read(tmp_im)
        tmpd_real[ifreq,:]=tmp_re[isrc-1,:]
        tmpd_imag[ifreq,:]=tmp_im[isrc-1,:]

    elif gather=='r' :
      ircv=inum
      tmpd_real=np.zeros((self.freqs.n,self.srcs.n),'f')
      tmpd_imag=np.zeros((self.freqs.n,self.srcs.n),'f')
      for ifreq in range(self.freqs.n):
        input_re.read(tmp_re)
        input_im.read(tmp_im)
        tmpd_real[ifreq,:]=tmp_re[:,ircv-1]
        tmpd_imag[ifreq,:]=tmp_im[:,ircv-1]

    elif gather=='f':
      ifreq=inum
      tmpd_real=np.zeros((self.srcs.n,self.rcvs.n),'f')
      tmpd_imag=np.zeros((self.srcs.n,self.rcvs.n),'f')
      for i in range(ifreq):
        input_re.read(tmpd_real)
        input_im.read(tmpd_imag)
    else: 
      tmpd_real=np.zeros((self.freqs.n,self.srcs.n,self.rcvs.n),'f')
      tmpd_imag=np.zeros((self.freqs.n,self.srcs.n,self.rcvs.n),'f')
      input_re.read(tmpd_real)
      input_im.read(tmpd_imag)

    self.d=tmpd_real+1.0j*tmpd_imag



  def write_header(self,fre,fim):
    self.outputre=rsf.Output(fre)
    self.outputim=rsf.Output(fim)
    self.outputre.put('n1',self.rcvs.n)
    self.outputre.put('n2',self.srcs.n)
    self.outputre.put('n3',self.freqs.n)
    self.outputre.put('d1',self.rcvs.d)
    self.outputre.put('d2',self.srcs.d)

    if self.freqs.n > 1 : 
      self.outputre.put('d3', ( self.freqs.d[1] - self.freqs.d[0] ))
      self.outputim.put('d3', ( self.freqs.d[1] - self.freqs.d[0] ))
    else :
      self.outputre.put('d3', self.freqs.d[0] )
      self.outputim.put('d3', self.freqs.d[0] )

    self.outputre.put('o1',self.rcvs.o)
    self.outputre.put('o2',self.srcs.o)
    self.outputre.put('o3',self.freqs.d[0])
    self.outputim.put('n1',self.rcvs.n)
    self.outputim.put('n2',self.srcs.n)
    #self.outputim.put('n3',self.freqs.n)
    self.outputim.put('d1',self.rcvs.d)
    self.outputim.put('d2',self.srcs.d)
    self.outputim.put('d3',self.freqs.d)
    self.outputim.put('o1',self.rcvs.o)
    self.outputim.put('o2',self.srcs.o)
    self.outputim.put('o3', self.freqs.d[0])
  
  def write(self,fre,fim):
    self.write_header(fre,fim)

    self.outputre.write(np.float32(np.real(self.d)))

    self.outputim.write(np.float32(np.imag(self.d)))

    self.write_close()

  def write_rsf_freq(self):
    self.outputre.write(np.float32(np.real(self.d)))
    self.outputim.write(np.float32(np.imag(self.d)))

  def write_close(self):
    self.outputre.close()
    self.outputim.close()

FreqDomainData = FreqdomainData
