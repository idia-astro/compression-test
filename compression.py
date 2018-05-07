
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
from PIL import Image
import os
import re
import itertools
import subprocess
#get_ipython().magic('matplotlib notebook')
#%cd "~"

# This will go in the notebook
ZFP_EXEC = "zfp"
SZ_EXEC = "sz"
TMP_DIR = "/tmp"
DATA_DIR = "~/data"

class Region:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_file(cls, filename):
        return cls(np.load(filename))
    
    @classmethod
    def _from_data(cls, data, stokes, channel, size, offset):
        if data.ndim == 4:
            data = data[stokes]
            
        if data.ndim == 3:
            data = data[channel]
                
        if size:
            ox, oy = offset
            w, h = size
                
            data = data[ox : ox + w, oy : oy + h]
            
        return cls(data.copy())
    
    @classmethod
    def from_hdf5(cls, filename, stokes=0, channel=0, size=None, offset=(0, 0)):
        with hdf5.File(self.hdf5filename, 'r') as f:
            return cls._from_data(f["0"]["DATA"], stokes, channel, size, offset)
            
    @classmethod
    def from_fits(cls, filename, stokes=0, channel=0, size=None, offset=(0, 0)):
        with fits.open(filename) as f:
            return cls._from_data(f[0].data, stokes, channel, size, offset)
    
    def write_to_file(self, filename):
        np.save(filename, self.data)
        
    def fix_nans(self, factor):
        X, Y = self.data.shape
        blurred = np.empty((X, Y))
        
        for i, j in itertools.product(range(0,X,factor), range(0,Y,factor)):
            blurred[i:i+2, j:j+2] = np.nanmean(self.data[i:i+2, j:j+2])

        self.data = np.where(np.isnan(self.data), blurred, self.data)
        
    def deltaErrors(self, other):
        delta = np.abs(self.data - other.data)
        delta = (~np.isnan(self.data)) * delta
        return np.nansum(delta), np.nanmax(delta)
    
    def image(self):
        pass # return colourmapped image


class Compressor:
    def compress(self, name, region, opts):
        raise NotImplemetedError()
        
class ZFP(Compressor):
    def compress(self, name, region, opts):
        region.write_to_file("original.arr")
        width, height = region.data.shape
        
        zip_p = subprocess.Popen(("zfp", "-i", "original.arr", "-2", width, height, "-f", opts, "-z", "-"), stdout=subprocess.PIPE)
        unzip_p = subprocess.check_output(("zfp", "-z", "-", "-2", width, height, "-f", opts ,"-o", "round_trip.arr"), stdin=zip_p.stdout)
        zip_p.wait()
        
        #...
    
    def compress_fixed_rate(self, region, rate):
        pass
    
    def compress_fixed_precision(self, region, precision):
        pass
    
    def compress_fixed_accuracy(self, region, tolerance):
        pass


class SZ(Compressor):
    def compress(self, name, region, opts):
        region.write_to_file("original.arr")
        width, height = region.data.shape
        # ...
    
    def compress_PSNR(self, region, PSNR):
        pass










# REPLACE WITH OBJECTS

def zfpCompress(inputArray, width, height, compressOpts):
    np.nan_to_num(inputArray).tofile("./tmpOrig.arr")
    output = get_ipython().getoutput('$ZFP_EXEC -i "./tmpOrig.arr" -2 $width $height -f $compressOpts -z - | $ZFP_EXEC -z - -2 $width $height -f $compressOpts -o "./tmpRoundtrip.arr"')
    m = re.search("zfp=(\d+)", output[0])
    compressedSize = int(m.group(1))
    roundtripArray = np.fromfile("./tmpRoundtrip.arr", 'f4')    
    get_ipython().system('rm "./tmpRoundtrip.arr"')
    get_ipython().system('rm "./tmpOrig.arr"')
    return roundtripArray, compressedSize

def zfpCompressFixedRate(inputArray, width, height, rate):
    return zfpCompress(inputArray, width, height, "-r {}".format(rate))

def zfpCompressFixedPrecision(inputArray, width, height, precision):
    return zfpCompress(inputArray, width, height, "-p {}".format(precision))

def zfpCompressFixedAccuracy(inputArray, width, height, tolerance):
    return zfpCompress(inputArray, width, height, "-a {}".format(tolerance))

def szCompress(inputArray, width, height, compressOpts):
    np.nan_to_num(inputArray).tofile("{}/tmpOrig.arr".format(TMP_DIR))
    output = get_ipython().getoutput('$SZ_EXEC -c sz.config $compressOpts -f -z -i "$TMP_DIR/tmpOrig.arr" -2 $width $height')
    output = get_ipython().getoutput('$SZ_EXEC -c sz.config $compressOpts -f -x -s "$TMP_DIR/tmpOrig.arr.sz" -2 $width $height        ')
    compressedSize = os.stat("{}/tmpOrig.arr.sz".format(TMP_DIR)).st_size
    roundtripArray = np.fromfile("{}/tmpOrig.arr.sz.out".format(TMP_DIR), 'f4')    
    get_ipython().system('rm "$TMP_DIR/tmpOrig.arr.sz"')
    get_ipython().system('rm "$TMP_DIR/tmpOrig.arr.sz.out"')
    get_ipython().system('rm "$TMP_DIR/tmpOrig.arr"')
    return roundtripArray, compressedSize

def szCompressPSNR(inputArray, width, height, PSNR):
    return szCompress(inputArray, width, height, "-M PSNR -S {}".format(PSNR))

#def deltaErrors(array1, array2):
    #deltaArray = np.abs(array1 - array2)
    #deltaArray = (~np.isnan(array1))*deltaArray
    #return np.nansum(deltaArray), np.nanmax(deltaArray)

#def nanFixed(arr, factor):    
    #downSampled = np.zeros([int(arr.shape[0]/factor),int(arr.shape[1]/factor)])
    #for i in range(downSampled.shape[0]):
        #for j in range(downSampled.shape[1]):
            #downSampled[i,j] = np.nanmean(arr[i*factor:i*factor+factor, j*factor:j*factor+factor])
    #resampledMeans = np.repeat(np.repeat(downSampled,factor, axis=0), factor, axis=1)
    #fixedArr = np.where(np.isnan(arr), resampledMeans, arr)
    #return fixedArr

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :, :3]


# REPLACE WITH FUNCTION CALLS


# GALFACTS N4 field (Q) without NaNs
originalFile = "./Downloads/n4_Q_no_nans.arr"
originalArray = np.fromfile(originalFile, 'f4')
sliceHeight = 1074
sliceWidth = 5850
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6

# GALFACTS N4 field (Q) without NaNs (512x512 slice)
originalFile = "./Downloads/n4_Q_no_nans.arr"
originalArray = np.fromfile(originalFile, 'f4')
originalArray = originalArray.reshape([-1, 5850])[768-256:768+256, 4600-256:4600+256].flatten()
sliceHeight = 512
sliceWidth = 512
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = -0.158637
scaleHigh = 0.120516

# GALFACTS N4 field (I) without NaNs
originalFile = "./Downloads/n4_no_nans.arr"
originalArray = np.fromfile(originalFile, 'f4')
sliceHeight = 1074
sliceWidth = 5850
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = 3.36162
scaleHigh = 8.11825

# GALFACTS N4 field (I) without NaNs (512x512 slice)
originalFile = "./Downloads/n4_no_nans.arr"
originalArray = np.fromfile(originalFile, 'f4')
originalArray = originalArray.reshape([-1, 5850])[768-256:768+256, 4600-256:4600+256].flatten()
sliceHeight = 512
sliceWidth = 512
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = 3.36162
scaleHigh = 8.11825

# full 67MB DEEP slice
originalFile = "./Downloads/full.arr"
originalArray = np.fromfile(originalFile, 'f4')
sliceHeight = 4096
sliceWidth = 4096
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = -0.000556335
scaleHigh = 0.000620103

# centre 512x512 slice subest
originalFile = "./Downloads/full.arr"
originalArray = np.fromfile(originalFile, 'f4')
originalArray = originalArray.reshape([-1, 4096])[2048-256:2048+256, 2048-256:2048+256].flatten()
sliceHeight = 512
sliceWidth = 512
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = -0.000556335
scaleHigh = 0.000620103

# supermosaic 512x512 slice subest
originalFile = "./Downloads/supermosaic.arr"
originalArray = np.fromfile(originalFile, 'f4')
originalArray = originalArray.reshape([-1, 4224])[900-256:900+256, 2000-256:2000+256].flatten()
sliceHeight = 512
sliceWidth = 512
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = -19.0277
scaleHigh = 66.8458

# supermosaic subset without NaNs
originalFile = "./Downloads/supermosaic.arr"
originalArray = np.fromfile(originalFile, 'f4')
originalArray = originalArray.reshape([-1, 4224])[100:-100, :].flatten()
sliceHeight = 1624
sliceWidth = 4224
fullSize = sliceHeight*sliceWidth*4
fullSizeMpix = fullSize/4e6
scaleLow = -19.0277
scaleHigh = 66.8458

originalArray.shape

sizesFixedRate = np.zeros(32)
absErrSumsFixedRate = np.zeros(32)
absErrMaxValsFixedRate = np.zeros(32)

for i in range(32):
    compressedArray, sizesFixedRate[i] = zfpCompressFixedRate(originalArray, sliceWidth, sliceHeight, i+1)
    absErrSumsFixedRate[i], absErrMaxValsFixedRate[i] = deltaErrors(originalArray, compressedArray)
    print ("{}...".format(i+1), end='')
print()

sizesFixedPrecision = np.zeros(32)
absErrSumsFixedPrecision = np.zeros(32)
absErrMaxValsFixedPrecision = np.zeros(32)

for i in range(32):
    compressedArray, sizesFixedPrecision[i] = zfpCompressFixedPrecision(originalArray, sliceWidth, sliceHeight, i+1)
    absErrSumsFixedPrecision[i], absErrMaxValsFixedPrecision[i] = deltaErrors(originalArray, compressedArray)
    print ("{}...".format(i+1), end='')
print()

accuracySettings = [10,9,8,7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,
                   5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]                   
numSettings = len(accuracySettings)
sizesFixedAccuracy = np.zeros(numSettings)
absErrSumsFixedAccuracy = np.zeros(numSettings)
absErrMaxValsFixedAccuracy = np.zeros(numSettings)

for i in range(numSettings):
    compressedArray, sizesFixedAccuracy[i] = zfpCompressFixedAccuracy(originalArray, sliceWidth, sliceHeight, accuracySettings[i])
    absErrSumsFixedAccuracy[i], absErrMaxValsFixedAccuracy[i] = deltaErrors(originalArray, compressedArray)
    print ("{} ({}/{})...".format(accuracySettings[i], i+1, numSettings), end='')
print()

sizesPSNR = np.zeros(32)
absErrSumsPSNR = np.zeros(32)
absErrMaxValsPSNR = np.zeros(32)

for i in range(32):
    PSNR = 70+i
    compressedArray, sizesPSNR[i] = szCompressPSNR(originalArray, sliceWidth, sliceHeight, PSNR)
    absErrSumsPSNR[i], absErrMaxValsPSNR[i] = deltaErrors(originalArray, compressedArray)
    print ("{} ({}/{})...".format(PSNR, (i+1), 32), end='')
print()

plt.close()
plt.scatter(sizesFixedRate*1e-6, absErrSumsFixedRate, label='ZFP (Fixed rate)')
plt.scatter(sizesFixedPrecision*1e-6, absErrSumsFixedPrecision, label='ZFP (Fixed precision)')
plt.scatter(sizesFixedAccuracy*1e-6, absErrSumsFixedAccuracy, label='ZFP (Fixed accuracy)')
plt.scatter(sizesPSNR*1e-6, absErrSumsPSNR, label='SZ (PSNR bounded)')
plt.legend()
plt.xlabel('Size (MB)')
plt.ylabel('Absolute error (sum)')

plt.close()
plt.scatter(sizesFixedRate*1e-6, absErrMaxValsFixedRate, label='ZFP (Fixed rate)')
plt.scatter(sizesFixedPrecision*1e-6, absErrMaxValsFixedPrecision, label='ZFP (Fixed precision)')
plt.scatter(sizesFixedAccuracy*1e-6, absErrMaxValsFixedAccuracy, label='ZFP (Fixed accuracy)')
plt.scatter(sizesPSNR*1e-6, absErrMaxValsPSNR, label='SZ (PSNR bounded)')
plt.legend()
plt.xlabel('Size (MB)')
plt.ylabel('Absolute error (max)')

#zfpCompressFixedPrecision(originalArray, sliceWidth, sliceHeight, 12)
zfpCompressFixedPrecision(np.nan_to_num(originalArray), sliceWidth, sliceHeight, 10)

reshapedArray = originalArray.reshape([-1, sliceWidth])
colormap = plt.cm.viridis
colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)
plt.imsave('{}/tmpOrig.png'.format(TMP_DIR), colormappedArray)
plt.close()
plt.imshow(colormappedArray)
originalImage = Image.open('{}/tmpOrig.png'.format(TMP_DIR))
originalImage = originalImage.convert("RGB")
pngSize = os.stat("{}/tmpOrig.png".format(TMP_DIR)).st_size
rawSize = sliceWidth*sliceHeight*4

sizesJPG = []
absErrSumsJPG = []
absErrMaxValsJPG = []

for i in range(60, 101):
    originalImage.save('{}/tmpCompressed.jpg'.format(TMP_DIR), format='JPEG', quality=i)
    sizesJPG.append(os.stat("{}/tmpCompressed.jpg".format(TMP_DIR)).st_size)
    compressedImageArray = plt.imread('{}/tmpCompressed.jpg'.format(TMP_DIR))
    deltaImage = np.abs(compressedImageArray/255.0-colormappedArray)
    absErrSumsJPG.append(np.nansum(deltaImage))
    absErrMaxValsJPG.append(np.nanmax(deltaImage))
    print ("{}...".format(i), end='')
print()

sizesJPG = np.array(sizesJPG)
absErrSumsJPG = np.array(absErrSumsJPG)
absErrMaxValsJPG = np.array(absErrMaxValsJPG)

accuracySettings = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,9,8,7,6,5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,
                   5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]                   
numSettings = len(accuracySettings)
sizesFixedAccuracy = np.zeros(numSettings)
absErrSumsImageFixedAccuracy = np.zeros(numSettings)
absErrMaxValsImageFixedAccuracy = np.zeros(numSettings)

reshapedArray = originalArray.reshape([-1, sliceWidth])
colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)

for i in range(numSettings):
    compressedArray, sizesFixedAccuracy[i] = zfpCompressFixedAccuracy(originalArray, sliceWidth, sliceHeight, accuracySettings[i])
    colormappedCompressedArray = get_colors(compressedArray.reshape([-1, sliceWidth]), colormap, scaleLow, scaleHigh)
    deltaImage = np.abs(colormappedCompressedArray-colormappedArray)
    absErrSumsImageFixedAccuracy[i] = np.nansum(deltaImage)
    absErrMaxValsImageFixedAccuracy[i] = np.nanmax(deltaImage)
    
    print ("{} ({}/{})...".format(accuracySettings[i], i+1, numSettings), end='')
print()

sizesPSNR = np.zeros(40)
absErrSumsImagePSNR = np.zeros(len(sizesPSNR))
absErrMaxValsImagePSNR = np.zeros(len(sizesPSNR))

reshapedArray = originalArray.reshape([-1, sliceWidth])
colormappedArray = get_colors(reshapedArray, colormap, scaleLow, scaleHigh)

for i in range(len(sizesPSNR)):
    PSNR = 60+i
    compressedArray, sizesPSNR[i] = szCompressPSNR(originalArray, sliceWidth, sliceHeight, PSNR)
    colormappedCompressedArray = get_colors(compressedArray.reshape([-1, sliceWidth]), colormap, scaleLow, scaleHigh)
    deltaImage = np.abs(colormappedCompressedArray-colormappedArray)
    absErrSumsImagePSNR[i] = np.nansum(deltaImage)
    absErrMaxValsImagePSNR[i] = np.nanmax(deltaImage)
    print ("{} ({}/{})...".format(PSNR, (i+1), len(sizesPSNR)), end='')
print()

plt.close()
plt.scatter(sizesJPG*1e-6, absErrSumsJPG/(sliceWidth*sliceHeight/100), marker='.', label='JPEG ({})'.format(colormap.name))
plt.scatter(sizesFixedAccuracy*1e-6, absErrSumsImageFixedAccuracy/(sliceWidth*sliceHeight/100), marker='.', label='ZFP ({})'.format(colormap.name))
plt.scatter(sizesPSNR*1e-6, absErrSumsImagePSNR/(sliceWidth*sliceHeight/100), marker='.', label='SZ ({})'.format(colormap.name))
plt.scatter(pngSize*1e-6, 0, marker='x', label='PNG ({})'.format(colormap.name))
plt.legend()
plt.xlabel('Size (MB)')
plt.ylabel('Absolute error (mean %)')
print("JPEG 95: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[-6]*1e-6, absErrSumsJPG[-6]/(sliceWidth*sliceHeight/100)))
print("JPEG 90: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[-11]*1e-6, absErrSumsJPG[-11]/(sliceWidth*sliceHeight/100)))
print("JPEG 60: {:0.3f} MB, {:0.3f}% mean error".format(sizesJPG[0]*1e-6, absErrSumsJPG[0]/(sliceWidth*sliceHeight/100)))

plt.close()
plt.scatter(sizesJPG*1e-6, absErrMaxValsJPG*100, marker='.', label='JPEG ({})'.format(colormap.name))
plt.scatter(sizesFixedAccuracy*1e-6, absErrMaxValsImageFixedAccuracy*100, marker='.', label='ZFP ({})'.format(colormap.name))
plt.scatter(sizesPSNR*1e-6, absErrMaxValsImagePSNR*100, marker='.', label='SZ ({})'.format(colormap.name))
plt.scatter(pngSize*1e-6, 0, marker='x', label='PNG ({})'.format(colormap.name))

plt.legend()
plt.xlabel('Size (MB)')
plt.ylabel('Absolute error (max %)')
print("JPEG 95: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[-6]*1e-6, absErrMaxValsJPG[-6]*100))
print("JPEG 90: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[-11]*1e-6, absErrMaxValsJPG[-11]*100))
print("JPEG 60: {:0.3f} MB, {:0.3f}% max error".format(sizesJPG[0]*1e-6, absErrMaxValsJPG[0]*100))

