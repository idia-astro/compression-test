
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
import collections
#get_ipython().magic('matplotlib notebook')
#%cd "~"

# This will go in the notebook
#ZFP_EXEC = "zfp"
#SZ_EXEC = "sz"
#TMP_DIR = "/tmp"
#DATA_DIR = "~/data"

ZSCALE = ZScaleInterval()

class Region:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_file(cls, filename, dtype, width, height):
        return cls(np.fromfile(filename, dtype).reshape((width, height)))
    
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
            
        return cls(data.copy())
    
    @classmethod
    def from_hdf5(cls, filename, stokes=0, channel=0, size=None, centre=(0, 0)):
        with hdf5.File(self.hdf5filename, 'r') as f:
            return cls._from_data(f["0"]["DATA"], stokes, channel, size, centre)
            
    @classmethod
    def from_fits(cls, filename, stokes=0, channel=0, size=None, centre=(0, 0)):
        with fits.open(filename) as f:
            return cls._from_data(f[0].data, stokes, channel, size, centre)
    
    def write_to_file(self, filename):
        self.data.tofile(filename)
        
    def fix_nans(self, factor):
        X, Y = self.data.shape
        blurred = np.empty((X, Y))
        
        for i, j in itertools.product(range(0,X,factor), range(0,Y,factor)):
            blurred[i:i+2, j:j+2] = np.nanmean(self.data[i:i+2, j:j+2])

        self.data = np.where(np.isnan(self.data), blurred, self.data)
        
    def delta_errors(self, other):
        delta = np.abs(self.data - other.data)
        delta = (~np.isnan(self.data)) * delta
        return np.nansum(delta), np.nanmax(delta)
    
    # TODO needs rewrite?
    # TODO: return another region?
    def get_colors(self, colormap):
        vmin, vmax = ZSCALE.get_limits(self.data)
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(self.data))[:, :, :3]

# TODO use temporary directory

def ZFP_compress(region, round_trip_filename, **kwargs):
    region.write_to_file("original.arr")
    width, height = region.data.shape
    
    zip_p = subprocess.run(("zfp", "-i", "original.arr", "-2", str(width), str(height), "-f", *kwargs['opts'], "-z", "-"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    unzip_p = subprocess.run(("zfp", "-z", "-", "-2", str(width), str(height), "-f", *kwargs['opts'] ,"-o", round_trip_filename), stdin=zip_p.stdout)
    zip_p.wait()
    
    m = re.search("zfp=(\d+)", zip_p.stderr) # ???!
    compressed_size = int(m.group(1))
    
    round_trip_region = Region.from_file(round_trip_filename, region.data.dtype, width, height)
    error_sum, error_max = region.delta_errors(round_trip_region)

    subprocess.run("rm", "original.arr")
    
    return round_trip_filename, (error_sum, error_max, compressed_size)

def ZFP_compress_fixed_rate(region, rate):
    return ZFP_compress(region, "ZFP_r_%s.arr" % rate, opts=("-r", str(rate)))

def ZFP_compress_fixed_precision(region, precision):
    return ZFP_compress(region, "ZFP_p_%s.arr" % precision, opts=("-p", str(precision)))

def ZFP_compress_fixed_accuracy(region, tolerance):
    return ZFP_compress(region, "ZFP_a_%s.arr" % tolerance, opts=("-a", str(tolerance)))


def SZ_compress(region, round_trip_filename, **kwargs):
    region.write_to_file("original.arr")
    width, height = region.data.shape
    
    zip_p = subprocess.run(("sz", "-c", "sz.config", *kwargs['opts'], "-f", "-z", "-i", "original.arr", "-2", str(width), str(height)))
    unzip_p = subprocess.run(("sz", "-c", "sz.config", *kwargs['opts'], "-f", "-x", "-s", "original.arr.sz", "-2", str(width), str(height)))
    
    compressed_size = os.stat("original.arr.sz").st_size
    roundtrip_array = Region.from_file("original.arr.sz.out", region.data.dtype, width, height)
    error_sum, error_max = region.delta_errors(round_trip_region)

    subprocess.run("rm", "original.arr")
    subprocess.run("rm", "original.arr.sz")
    subprocess.run("mv", "original.arr.sz.out", round_trip_filename)
    
    return round_trip_filename, (error_sum, error_max, compressed_size)
    
def SZ_compress_PSNR(region, PSNR):
    return SZ_compress(region, "SZ_PSNR_%s.arr" % PSNR, opts=("-M", "PSNR", "-S", str(PSNR)))

def JPG_compress_quality(region, round_trip_filename, **kwargs):
    region.data.save(round_trip_filename, format='JPEG', quality=kwargs['quality'])
    compressed_size = os.stat(round_trip_filename).st_size
    roundtrip_array = Region.from_file(round_trip_filename, region.data.dtype, width, height)
    #deltaImage = np.abs(compressedImageArray/255.0-colormappedArray)
    #absErrSumsJPG.append(np.nansum(deltaImage))
    #absErrMaxValsJPG.append(np.nanmax(deltaImage))
    return round_trip_filename, (error_sum, error_max, compressed_size)

def JPG_compress_quality(region, quality):
    return JPG_compress(region, "JPG_q_%s.arr" % quality, quality=quality)


# TODO: add colourmap to everything; return colourmapped deltas as well (only colourmapped for jpeg)
def compare_algorithm(region, function, parameter_range, colourmap):
    return dict(function(region, colourmap, p) for p in parameter_range)

# TODO: pass in ranges and working directory?
# TODO and colormap
# TODO split out
#def compare_algorithms(region,):
    #results = {}
    
    #for function, params in (
        #(ZFP_compress_fixed_rate, range(1, 32+1)),
        #(ZFP_compress_fixed_precision, range(1, 32+1)),
        #(ZFP_compress_fixed_accuracy, list(range(1, 21)) + [
                #0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                #5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 
                #5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7
            #]
        #),
        #(SZ_compress_PSNR, range(60, 100)),
        #(JPG_compress_quality, range(60, 101)),
    #):
        #results.update(compare_parameters(region, function, params))
    
    #return results
    

def clean_up():
    pass # delete all of the temporary files

# TODO: merge comparisons







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

