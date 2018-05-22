
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
    
    def write_to_file(self, filename):
        self.data.tofile(filename)
        
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
        delta = (~np.isnan(self.image)) * delta
        return np.nansum(delta), np.nanmax(delta)

# TODO use temporary directory; make a class for this?
# instantiate it; add properties for data directory, colourmap, etc?

def ZFP_compress(region, image, *args):
    width, height = region.data.shape
    
    zip_p = subprocess.Popen(("zfp", "-i", "original.arr", "-2", str(width), str(height), "-f", *args, "-z", "-"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    unzip_p = subprocess.Popen(("zfp", "-z", "-", "-2", str(width), str(height), "-f", *args ,"-o", "round_trip.arr"), stdin=zip_p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    zip_p.stdout.close()
    
    m = re.search("zfp=(\d+)", unzip_p.communicate()[1].decode())
    compressed_size = int(m.group(1))
    
    round_trip_region = Region.from_file("round_trip.arr", "f4", width, height)
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
    
    subprocess.run(("sz", "-c", "sz.config", *args, "-f", "-z", "-i", "original.arr", "-2", str(width), str(height)))
    subprocess.run(("sz", "-c", "sz.config", *args, "-f", "-x", "-s", "original.arr.sz", "-2", str(width), str(height)))
    
    compressed_size = os.stat("original.arr.sz").st_size
    round_trip_region = Region.from_file("original.arr.sz.out", "f4", width, height)

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
    
    raw_errors = {}
    image_errors = {}
    size_fractions = {}
    
    region.write_to_file("original.arr")
    original_raw_size = os.stat("original.arr").st_size
    
    image = region.colourmapped(colourmap)
    image.to_png("original.png")
    original_image_size = os.stat("original.png").st_size
    
    for label, function, params in ALGORITHMS:
        for p in params:
            key = (label, p)
            
            round_trip_region, compressed_image, compressed_raw_size, compressed_image_size = function(region, image, p)
            
            if round_trip_region:
                raw_errors[key] = region.delta_errors(round_trip_region)
            
            image_errors[key] = image.delta_errors(compressed_image)
            
            if compressed_raw_size:
                size_fractions[key] = compressed_raw_size/original_raw_size
            elif compressed_image_size:
                size_fractions[key] = compressed_image_size/original_image_size
                
    
    subprocess.run(("rm", "original.arr"))
    subprocess.run(("rm", "original.png"))

    return raw_errors, image_errors, size_fractions

# TODO TODO TODO

# split out results for different algorithms
# maybe put this in a pandas table?
# scatter plot for raw error
# scatter plot for image error
# plot of size vs error
# plot of error vs size
# put these in one figure so that function can be called from one notebook cell
# visualisation of image error: how? Display original plus compressed versions, with a slider to adjust the size?





# TODO: rewrite to use new data; put in function; use from notebook

#plt.close()
#plt.scatter(sizesFixedRate*1e-6, absErrSumsFixedRate, label='ZFP (Fixed rate)')
#plt.scatter(sizesFixedPrecision*1e-6, absErrSumsFixedPrecision, label='ZFP (Fixed precision)')
#plt.scatter(sizesFixedAccuracy*1e-6, absErrSumsFixedAccuracy, label='ZFP (Fixed accuracy)')
#plt.scatter(sizesPSNR*1e-6, absErrSumsPSNR, label='SZ (PSNR bounded)')
#plt.legend()
#plt.xlabel('Size (MB)')
#plt.ylabel('Absolute error (sum)')

#plt.close()
#plt.scatter(sizesFixedRate*1e-6, absErrMaxValsFixedRate, label='ZFP (Fixed rate)')
#plt.scatter(sizesFixedPrecision*1e-6, absErrMaxValsFixedPrecision, label='ZFP (Fixed precision)')
#plt.scatter(sizesFixedAccuracy*1e-6, absErrMaxValsFixedAccuracy, label='ZFP (Fixed accuracy)')
#plt.scatter(sizesPSNR*1e-6, absErrMaxValsPSNR, label='SZ (PSNR bounded)')
#plt.legend()
#plt.xlabel('Size (MB)')
#plt.ylabel('Absolute error (max)')







##zfpCompressFixedPrecision(originalArray, sliceWidth, sliceHeight, 12)
#zfpCompressFixedPrecision(np.nan_to_num(originalArray), sliceWidth, sliceHeight, 10)

#reshapedArray = originalArray.reshape([-1, sliceWidth])
#colormap = plt.cm.viridis
#colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)
#plt.imsave('{}/tmpOrig.png'.format(TMP_DIR), colormappedArray)
#plt.close()
#plt.imshow(colormappedArray)
#originalImage = Image.open('{}/tmpOrig.png'.format(TMP_DIR))
#originalImage = originalImage.convert("RGB")
#pngSize = os.stat("{}/tmpOrig.png".format(TMP_DIR)).st_size
#rawSize = sliceWidth*sliceHeight*4

#sizesJPG = []
#absErrSumsJPG = []
#absErrMaxValsJPG = []

#for i in range(60, 101):
    #originalImage.save('{}/tmpCompressed.jpg'.format(TMP_DIR), format='JPEG', quality=i)
    #sizesJPG.append(os.stat("{}/tmpCompressed.jpg".format(TMP_DIR)).st_size)
    #compressedImageArray = plt.imread('{}/tmpCompressed.jpg'.format(TMP_DIR))
    #deltaImage = np.abs(compressedImageArray/255.0-colormappedArray)
    #absErrSumsJPG.append(np.nansum(deltaImage))
    #absErrMaxValsJPG.append(np.nanmax(deltaImage))
    #print ("{}...".format(i), end='')
#print()

#sizesJPG = np.array(sizesJPG)
#absErrSumsJPG = np.array(absErrSumsJPG)
#absErrMaxValsJPG = np.array(absErrMaxValsJPG)

#accuracySettings = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,9,8,7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,
                   #5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]                   
#numSettings = len(accuracySettings)
#sizesFixedAccuracy = np.zeros(numSettings)
#absErrSumsImageFixedAccuracy = np.zeros(numSettings)
#absErrMaxValsImageFixedAccuracy = np.zeros(numSettings)

#reshapedArray = originalArray.reshape([-1, sliceWidth])
#colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)

#for i in range(numSettings):
    #compressedArray, sizesFixedAccuracy[i] = zfpCompressFixedAccuracy(originalArray, sliceWidth, sliceHeight, accuracySettings[i])
    #colormappedCompressedArray = get_colors(compressedArray.reshape([-1, sliceWidth]), colormap, scaleLow, scaleHigh)
    #deltaImage = np.abs(colormappedCompressedArray-colormappedArray)
    #absErrSumsImageFixedAccuracy[i] = np.nansum(deltaImage)
    #absErrMaxValsImageFixedAccuracy[i] = np.nanmax(deltaImage)
    
    #print ("{} ({}/{})...".format(accuracySettings[i], i+1, numSettings), end='')
#print()

#sizesPSNR = np.zeros(40)
#absErrSumsImagePSNR = np.zeros(len(sizesPSNR))
#absErrMaxValsImagePSNR = np.zeros(len(sizesPSNR))

#reshapedArray = originalArray.reshape([-1, sliceWidth])
#colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)

#for i in range(len(sizesPSNR)):
    #PSNR = 60+i
    #compressedArray, sizesPSNR[i] = szCompressPSNR(originalArray, sliceWidth, sliceHeight, PSNR)
    #colormappedCompressedArray = get_colors(compressedArray.reshape([-1, sliceWidth]), colormap, scaleLow, scaleHigh)
    #deltaImage = np.abs(colormappedCompressedArray-colormappedArray)
    #absErrSumsImagePSNR[i] = np.nansum(deltaImage)
    #absErrMaxValsImagePSNR[i] = np.nanmax(deltaImage)
    #print ("{} ({}/{})...".format(PSNR, (i+1), len(sizesPSNR)), end='')
#print()









#plt.close()
#plt.scatter(sizesJPG*1e-6, absErrSumsJPG/(sliceWidth*sliceHeight/100), marker='.', label='JPEG ({})'.format(colormap.name))
#plt.scatter(sizesFixedAccuracy*1e-6, absErrSumsImageFixedAccuracy/(sliceWidth*sliceHeight/100), marker='.', label='ZFP ({})'.format(colormap.name))
#plt.scatter(sizesPSNR*1e-6, absErrSumsImagePSNR/(sliceWidth*sliceHeight/100), marker='.', label='SZ ({})'.format(colormap.name))
#plt.scatter(pngSize*1e-6, 0, marker='x', label='PNG ({})'.format(colormap.name))
#plt.legend()
#plt.xlabel('Size (MB)')
#plt.ylabel('Absolute error (mean %)')
#print("JPEG 95: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[-6]*1e-6, absErrSumsJPG[-6]/(sliceWidth*sliceHeight/100)))
#print("JPEG 90: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[-11]*1e-6, absErrSumsJPG[-11]/(sliceWidth*sliceHeight/100)))
#print("JPEG 60: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[0]*1e-6, absErrSumsJPG[0]/(sliceWidth*sliceHeight/100)))

#plt.close()
#plt.scatter(sizesJPG*1e-6, absErrMaxValsJPG*100, marker='.', label='JPEG ({})'.format(colormap.name))
#plt.scatter(sizesFixedAccuracy*1e-6, absErrMaxValsImageFixedAccuracy*100, marker='.', label='ZFP ({})'.format(colormap.name))
#plt.scatter(sizesPSNR*1e-6, absErrMaxValsImagePSNR*100, marker='.', label='SZ ({})'.format(colormap.name))
#plt.scatter(pngSize*1e-6, 0, marker='x', label='PNG ({})'.format(colormap.name))

#plt.legend()
#plt.xlabel('Size (MB)')
#plt.ylabel('Absolute error (max %)')
#print("JPEG 95: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[-6]*1e-6, absErrMaxValsJPG[-6]*100))
#print("JPEG 90: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[-11]*1e-6, absErrMaxValsJPG[-11]*100))
#print("JPEG 60: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[0]*1e-6, absErrMaxValsJPG[0]*100))

