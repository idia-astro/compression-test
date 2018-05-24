
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
        delta = np.abs(self.data - other.data) / np.abs(self.data)
        delta = (~np.isnan(self.data)) * delta
        return np.nanmean(delta), np.nanmax(delta)
    
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
        delta = np.abs(self.image - other.image) / np.abs(self.image)
        delta = (~np.isnan(np.abs(self.image))) * delta
        return np.nanmean(delta), np.nanmax(delta)

# TODO use temporary directory and executable paths


class Compressor:
    
    def __init__(self, region, image, temp_dir=".", zfp="zfp", sz="sz"):
        self.region = region
        self.image = image
        
        self.temp_dir = os.path.abspath(temp_dir)
        self.zfp = zfp
        self.sz = sz

    def ZFP_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        width, height = self.region.data.shape
        
        self.region.write_to_file("original.arr", "f4")
        
        zip_p = subprocess.Popen((self.zfp, "-i", "original.arr", "-2", str(width), str(height), "-f", *args, "-z", "-"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        unzip_p = subprocess.Popen((self.zfp, "-z", "-", "-2", str(width), str(height), "-f", *args ,"-o", "round_trip.arr"), stdin=zip_p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        zip_p.stdout.close()
        
        m = re.search("zfp=(\d+)", unzip_p.communicate()[1].decode())
        compressed_size = int(m.group(1))
        
        round_trip_region = Region.from_file("round_trip.arr", "f4", width, height)
        
        subprocess.run(("rm", "original.arr"))
        subprocess.run(("rm", "round_trip.arr"))
        
        compressed_image = self.image.clone_colourmap_to(round_trip_region)
        
        os.chdir(prev)
        return round_trip_region, compressed_image, compressed_size, None

    def ZFP_compress_fixed_rate(self, rate):
        return self.ZFP_compress("-r", str(rate))

    def ZFP_compress_fixed_precision(self, precision):
        return self.ZFP_compress("-p", str(precision))

    def ZFP_compress_fixed_accuracy(self, tolerance):
        return self.ZFP_compress("-a", str(tolerance))

    def SZ_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        width, height = self.region.data.shape
            
        self.region.write_to_file("original.arr", "f4")
        
        subprocess.run((self.sz, "-c", "sz.config", *args, "-f", "-z", "-i", "original.arr", "-2", str(width), str(height)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run((self.sz, "-c", "sz.config", *args, "-f", "-x", "-s", "original.arr.sz", "-2", str(width), str(height)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        compressed_size = os.stat("original.arr.sz").st_size
        round_trip_region = Region.from_file("original.arr.sz.out", "f4", width, height)

        subprocess.run(("rm", "original.arr"))
        subprocess.run(("rm", "original.arr.sz"))
        subprocess.run(("rm", "original.arr.sz.out"))
        
        compressed_image = self.image.clone_colourmap_to(round_trip_region)
        
        os.chdir(prev)
        return round_trip_region, compressed_image, compressed_size, None
        
    def SZ_compress_PSNR(self, PSNR):
        return self.SZ_compress("-M", "PSNR", "-S", str(PSNR))

    def JPG_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.image.to_jpg("compressed.jpg", args[0])
        
        compressed_size = os.stat("compressed.jpg").st_size
        compressed_image = ColourmappedRegion.from_jpg("compressed.jpg")

        subprocess.run(("rm", "compressed.jpg"))
        
        os.chdir(prev)
        return None, compressed_image, None, compressed_size

    def JPG_compress_quality(self, quality):
        return self.JPG_compress(quality)


class Comparator:
    def __init__(self, results, compressor):
        self.results = results
        self.compressor = compressor

    ALGORITHMS = (
        ("ZFP (Fixed rate)", "ZFP_compress_fixed_rate", range(1, 32+1)),
        ("ZFP (Fixed precision)", "ZFP_compress_fixed_precision", range(1, 32+1)),
        ("ZFP (Fixed accuracy)", "ZFP_compress_fixed_accuracy", list(range(1, 21)) + [
                    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                    5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 
                    5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7
                ]
            ),
        ("SZ (PSNR bounded)", "SZ_compress_PSNR", range(60, 100)),
        ("JPEG", "JPG_compress_quality", range(60, 101)),
    )

    @classmethod
    def compare_algorithms(cls, region, colourmap, temp_dir=".", zfp="zfp", sz="sz"):
        results = []
        
        original_raw_size = np.dtype("f4").itemsize * region.data.size
        
        image = region.colourmapped(colourmap)
        image.to_png("original.png")
        original_image_size = os.stat("original.png").st_size
        
        compressor = Compressor(region, image, temp_dir, zfp, sz)
                
        for label, function_name, params in cls.ALGORITHMS:
            
            for p in params:
                round_trip_region, compressed_image, compressed_raw_size, compressed_image_size = getattr(compressor, function_name)(p)
                
                if round_trip_region:
                    raw_error_mean, raw_error_max = region.delta_errors(round_trip_region)
                else:
                    raw_error_mean, raw_error_max = None, None
                
                image_error_mean, image_error_max = image.delta_errors(compressed_image)
                
                if compressed_raw_size:
                    size_fraction = compressed_raw_size/original_raw_size
                elif compressed_image_size:
                    size_fraction = compressed_image_size/original_image_size
                    
                results.append({
                    "label": label,
                    "function_name": function_name, # may need it later to regenerate images
                    "param": p, # may need it later to regenerate images
                    "raw_error_mean": raw_error_mean, 
                    "raw_error_max": raw_error_max, 
                    "image_error_mean": image_error_mean, 
                    "image_error_max": image_error_max, 
                    "size_fraction": size_fraction,
                })

                    
        subprocess.run(("rm", "original.png"))
        
        return cls(results, compressor)

    # insufficiently generic; just inline these in the plot function
    def get(self, fields, where):
        return [[r[f] for f in fields] for r in self.results if all(r[k] == v for k, v in where.items())]
    
    def unique(self, field):
        return {r[field] for r in self.results}
        
    def plot(self, xfield, yfield, xlabel, ylabel):

        plt.close()
        
        for label in self.unique("label"):
            xy = sorted(self.get((xfield, yfield), {"label": label})) # sort by xfield
            x, y = zip(*xy)
            
            if all(v is None for v in y):
                continue # skip e.g. non-existent raw errors for JPEG
            
            plt.plot(x, y, marker='o', ls='', label=label)
            
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    def show_images(self, size_min, size_max):
        # TODO: get all images within a size range and display them together with original image. Stacked? Raise on mouseover?
