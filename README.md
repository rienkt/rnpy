# rnpy

A Python library for seismic data processing and modeling.

## Overview

**rnpy** provides tools for analyzing, processing, and simulating seismic waveform data, with a focus on seismic imaging, signal processing, and geophysical data management.

## Features

- **Signal Processing** – Fourier transforms, wavelet analysis, convolution, deconvolution, filtering, F-K (frequency-wavenumber) filtering, normal moveout (NMO) corrections, cross-correlation, and K-K domain operations
- **Source/Receiver Management** – Active source (controlled seismic) and passive/continuous seismic station handling with spatial coordinate support
- **Data I/O** – RSF/Madagascar format support and HDF5 storage
- **Visualization** – Wiggle trace plots, F-K domain plots, and focal mechanism (beach ball) diagrams

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/rienkt/rnpy.git
cd rnpy
pip install numpy scipy matplotlib pandas
```

## Project Structure

```
rn/
├── libs/       # Core signal processing algorithms
├── rsf/        # RSF (Regular Sampled Format) data classes
├── bin/        # Source/receiver and station management
│   ├── active/       # Active source seismic
│   ├── continuous/   # Passive/continuous seismic monitoring
│   └── model/        # Model construction utilities
├── coord/      # Coordinate system and offset calculations
├── model/      # Model creation and modification (for use in forward modeling or inversion)
├── plot/       # Matplotlib-based visualization tools
├── hdf5/       # HDF5 file I/O
└── win/        # Window functions and tapering
```

## Usage

```python
import rn

# F-K domain filtering
from rn.libs import fk
filtered = fk.rn_fk(data, dt, dx)

# Working with 2D seismic models (RSF format)
from rn.rsf.model import Model
model = Model(nx=100, nz=100, dx=10, dz=10)

# Active source location management
from rn.bin.active.core import rn_loc
locations = rn_loc(n=50)

# Wiggle trace plot
from rn.plot import plot
plot.rn_plot_wiggle(data, dt)
```

## Dependencies

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/) *(optional)*
- [h5py](https://www.h5py.org/) *(optional, for HDF5 support)*
