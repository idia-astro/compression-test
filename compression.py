
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import h5py
from PIL import Image
import os
import re
import itertools
import subprocess

# This will go in the notebook
#ZFP_EXEC = "zfp"
#SZ_EXEC = "sz"
#TMP_DIR = "/tmp"
#DATA_DIR = "~/data"


class Region:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_file(cls, filename, dtype, width, height):
        return cls(np.fromfile(filename, dtype=dtype).reshape(width, height))
    
    @staticmethod
    def _fix_nans(data, factor):
        X, Y = data.shape
        blurred = np.empty((X, Y))
        
        for i, j in itertools.product(range(0,X,factor), range(0,Y,factor)):
            blurred[i:i+2, j:j+2] = np.nanmean(data[i:i+2, j:j+2])

        return np.where(np.isnan(data), blurred, data)
    
    @classmethod
    def _from_data(cls, data, stokes, channel, size, centre):
        if data.ndim == 4:
            data = data[stokes]
            
        if data.ndim == 3:
            data = data[channel]
                
        if size:
            w, h = size
            w_2, h_2 = int(w/2), int(h/2)
            cx, cy = centre
                
            data = data[cx - w_2 : cx + w_2, cy - h_2 : cy + h_2]
            
        data = cls._fix_nans(data, 2)
            
        return cls(data)
    
    @classmethod
    def from_hdf5(cls, filename, stokes=0, channel=0, size=None, centre=(0, 0)):
        with h5py.File(filename, 'r') as f:
            return cls._from_data(f["0"]["DATA"], stokes, channel, size, centre)
            
    @classmethod
    def from_fits(cls, filename, stokes=0, channel=0, size=None, centre=(0, 0)):
        with fits.open(filename) as f:
            return cls._from_data(f[0].data, stokes, channel, size, centre)
    
    def write_to_file(self, filename, dtype):
        self.data.astype(dtype).tofile(filename)
        
    def delta_errors(self, other):
        delta = np.abs(self.data - other.data)
        delta = (~np.isnan(self.data)) * delta
        return np.nansum(delta), np.nanmax(delta)
    
    def colourmapped(self, colourmap, vmin=None, vmax=None):
        return ColourmappedRegion.from_data(self.data, colourmap, vmin, vmax)
    


class ColourmappedRegion:
    
    ZSCALE = ZScaleInterval()
    
    def __init__(self, image, colourmap=None, vmin=None, vmax=None):
        self.image = image
        self.colourmap = colourmap
        self.vmin = vmin
        self.vmax = vmax
    
    @classmethod
    def from_data(cls, data, colourmap, vmin, vmax):
        if vmin is None or vmax is None:
            vmin, vmax = cls.ZSCALE.get_limits(data)
        norm = plt.Normalize(vmin, vmax)
        return cls(colourmap(norm(data))[:, :, :3], colourmap, vmin, vmax)
    
    @classmethod
    def from_png(cls, filename):
        return cls(plt.imread(filename))
    
    @classmethod
    def from_jpg(cls, filename):
        return cls(plt.imread(filename)/255)
    
    def to_png(self, filename):
        plt.imsave(filename, self.image)
        
    def to_jpg(self, filename, quality):
        Image.fromarray((self.image*255).astype(np.uint8)).save(filename, format='JPEG', quality=quality)
        
    def clone_colourmap_to(self, region):
        return ColourmappedRegion.from_data(region.data, self.colourmap, self.vmin, self.vmax)
        
    def delta_errors(self, other):
        delta = np.abs(self.image - other.image)
        delta = (~np.isnan(np.abs(self.image))) * delta
        return np.nansum(delta), np.nanmax(delta)

# TODO use temporary directory; make a class for this?
# instantiate it; add properties for data directory, colourmap, etc?

def ZFP_compress(region, image, *args):
    width, height = region.data.shape
    
    region.write_to_file("original.arr", "f4")
    
    zip_p = subprocess.Popen(("zfp", "-i", "original.arr", "-2", str(width), str(height), "-f", *args, "-z", "-"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    unzip_p = subprocess.Popen(("zfp", "-z", "-", "-2", str(width), str(height), "-f", *args ,"-o", "round_trip.arr"), stdin=zip_p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    zip_p.stdout.close()
    
    m = re.search("zfp=(\d+)", unzip_p.communicate()[1].decode())
    compressed_size = int(m.group(1))
    
    round_trip_region = Region.from_file("round_trip.arr", "f4", width, height)
    
    subprocess.run(("rm", "original.arr"))
    subprocess.run(("rm", "round_trip.arr"))
    
    compressed_image = image.clone_colourmap_to(round_trip_region)
    
    return round_trip_region, compressed_image, compressed_size, None

def ZFP_compress_fixed_rate(region, image, rate):
    return ZFP_compress(region, image, "-r", str(rate))

def ZFP_compress_fixed_precision(region, image, precision):
    return ZFP_compress(region, image, "-p", str(precision))

def ZFP_compress_fixed_accuracy(region, image, tolerance):
    return ZFP_compress(region, image, "-a", str(tolerance))


def SZ_compress(region, image, *args):
    width, height = region.data.shape
        
    region.write_to_file("original.arr", "f4")
    
    subprocess.run(("sz", "-c", "sz.config", *args, "-f", "-z", "-i", "original.arr", "-2", str(width), str(height)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(("sz", "-c", "sz.config", *args, "-f", "-x", "-s", "original.arr.sz", "-2", str(width), str(height)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    compressed_size = os.stat("original.arr.sz").st_size
    round_trip_region = Region.from_file("original.arr.sz.out", "f4", width, height)

    subprocess.run(("rm", "original.arr"))
    subprocess.run(("rm", "original.arr.sz"))
    subprocess.run(("rm", "original.arr.sz.out"))
    
    compressed_image = image.clone_colourmap_to(round_trip_region)
    
    return round_trip_region, compressed_image, compressed_size, None
    
def SZ_compress_PSNR(region, image, PSNR):
    return SZ_compress(region, image, "-M", "PSNR", "-S", str(PSNR))

def JPG_compress(region, image, *args):
    image.to_jpg("compressed.jpg", args[0])
    
    compressed_size = os.stat("compressed.jpg").st_size
    compressed_image = ColourmappedRegion.from_jpg("compressed.jpg")

    subprocess.run(("rm", "compressed.jpg"))
    
    return None, compressed_image, None, compressed_size

def JPG_compress_quality(region, image, quality):
    return JPG_compress(region, image, quality)


# TODO: regenerate the regions when we want to see one; maybe cache some later
# TODO: how do we dynamically load the function? Keep a dictionary mapping of name to function? Put them all in a class so we can just getattr?

ALGORITHMS = (
    ("ZFP (Fixed rate)", ZFP_compress_fixed_rate, range(1, 32+1)),
    ("ZFP (Fixed precision)", ZFP_compress_fixed_precision, range(1, 32+1)),
    ("ZFP (Fixed accuracy)", ZFP_compress_fixed_accuracy, list(range(1, 21)) + [
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 
                5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7
            ]
        ),
    ("SZ (PSNR bounded)", SZ_compress_PSNR, range(60, 100)),
    ("JPEG", JPG_compress_quality, range(60, 101)),
)

def compare_algorithms(region, colourmap):
    results = {}
    
    original_raw_size = np.dtype("f4").itemsize * region.data.size
    
    image = region.colourmapped(colourmap)
    image.to_png("original.png")
    original_image_size = os.stat("original.png").st_size
    
    for label, function, params in ALGORITHMS:
        results[label] = {}
        
        for p in params:
            round_trip_region, compressed_image, compressed_raw_size, compressed_image_size = function(region, image, p)
            
            if round_trip_region:
                raw_e_sum, raw_e_max = region.delta_errors(round_trip_region)
            
            image_e_sum, image_r_max = image.delta_errors(compressed_image)
            
            if compressed_raw_size:
                size_fraction = compressed_raw_size/original_raw_size
            elif compressed_image_size:
                size_fraction = compressed_image_size/original_image_size
                
            results[label][p] = (raw_e_sum, raw_e_max, image_e_sum, image_r_max, size_fraction)
                
    subprocess.run(("rm", "original.png"))

    return results
