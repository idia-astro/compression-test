
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import h5py
from PIL import Image
import os
import re
import itertools
import subprocess
import operator

from ipywidgets import interact, Checkbox, FloatSlider, IntSlider, IntText, SelectMultiple, Text, Dropdown, Select, fixed


class DataWrapperMixin:
    def _delta_errors(self, other, *functions, **kwargs):
        self_data = getattr(self, self.DATANAME)
        other_data = getattr(other, self.DATANAME)
        
        mask = kwargs.get("mask", None)
        if mask is not None:
            self_data = self_data[mask]
            other_data = other_data[mask]
        
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)
            delta = np.abs(self_data - other_data)
            
            absolute_errors = [f(delta) for f in functions]
            
            # relative error
            delta = delta / np.abs(self_data)
            # remove infs from division by zero
            delta = delta[~np.isinf(delta)]
            
            relative_errors = [f(delta) for f in functions]
        
        # these functions should exclude nans -- nanmean, nanmax, etc.
        return tuple(zip(absolute_errors, relative_errors))
    
    @classmethod
    def from_tiles(cls, tiles, shape):
        x, y = shape[:2]
        data = np.empty(shape)
        
        tile_size = getattr(tiles[0], cls.DATANAME).shape[0]
        
        for i in range(0, x, tile_size):
            for j in range(0, y, tile_size):
                data[i : i + tile_size, j : j + tile_size] = getattr(tiles.pop(0), cls.DATANAME)
                
        return cls(data)
    
    def _tiles(self, tile_size):
        data = getattr(self, self.DATANAME)
        cls = self.__class__
        x, y = data.shape[:2]
        
        for i in range(0, x, tile_size):
            for j in range(0, y, tile_size):
                yield cls(data[i : i + tile_size, j : j + tile_size])


class Region(DataWrapperMixin):
    DATANAME = "data"
    
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_file(cls, filename, dtype, width, height):
        return cls(np.fromfile(filename, dtype=dtype).reshape(width, height))
    
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
        
    def delta_errors(self, other, *functions, **kwargs):
        return self._delta_errors(other, *functions, **kwargs)
    
    def colourmapped(self, colourmap, vmin=None, vmax=None, log=False):
        return ColourmappedRegion.from_data(self.data, colourmap, vmin, vmax, log)
    
    def tiles(self, tile_size):
        return self._tiles(tile_size)


class ColourmappedRegion(DataWrapperMixin):
    DATANAME = "image"
    ZSCALE = ZScaleInterval()
    
    def __init__(self, image, colourmap=None, vmin=None, vmax=None, log=False):
        self.image = image
        self.colourmap = colourmap
        self.vmin = vmin
        self.vmax = vmax
        self.log = log
    
    @classmethod
    def from_data(cls, data, colourmap, vmin, vmax, log):
        if vmin is None or vmax is None:
            vmin, vmax = cls.ZSCALE.get_limits(data)
            if log:
                vmax = np.max(data)
        norm_data = Normalize(vmin, vmax)(data)
        if log:
            a = 1000            
            norm_data = np.log(a * norm_data + 1)/np.log(a)
        return cls(colourmap(norm_data)[:, :, :3], colourmap, vmin, vmax, log)
    
    @classmethod
    def from_png(cls, filename):
        return cls(plt.imread(filename)[:, :, :3])
    
    @classmethod
    def from_jpg(cls, filename):
        return cls(plt.imread(filename)/255)
    
    def to_png(self, filename):
        plt.imsave(filename, self.image)
        
    def to_jpg(self, filename, quality):
        Image.fromarray((self.image*255).astype(np.uint8)).save(filename, format='JPEG', quality=quality)
        
    def clone_colourmap_to(self, region):
        return ColourmappedRegion.from_data(region.data, self.colourmap, self.vmin, self.vmax, self.log)
        
    def delta_errors(self, other, *functions, **kwargs):
        return self._delta_errors(other, *functions, **kwargs)
    
    def tiles(self, tile_size):
        return self._tiles(tile_size)
    
    def display(self):
        Image.fromarray((self.image*255).astype(np.uint8)).show()


def fix_nans(data, method):
    if np.all(np.isnan(data)):
        raise ValueError("This channel contains only NaNs! Please select a different channel.")
    
    print("Number of NaNs (before interpolation):", np.sum(np.isnan(data)))
    
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    data = np.ma.masked_invalid(data)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~data.mask]
    y1 = yy[~data.mask]
    new_data = data[~data.mask]
    
    data = interpolate.griddata((x1, y1), new_data.ravel(), (xx, yy), method=method)
    
    print("Number of NaNs (after interpolation):", np.sum(np.isnan(data)))

    return data


class Compressor:
    
    def __init__(self, region, image, temp_dir=".", zfp="zfp", sz="sz", bpgenc="bpgenc", bpgdec="bpgdec", tile_size=256):
        self.region = region
        self.image = image
        
        self.temp_dir = os.path.abspath(temp_dir)
        self.zfp = zfp
        self.sz = sz
        self.bpgenc = bpgenc
        self.bpgdec = bpgdec
        
        self.tile_size = tile_size

    def ZFP_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        round_trip_tiles = []
        compressed_size = 0
                
        for i, tile in enumerate(self.region.tiles(self.tile_size)):
            orig_name = "original_%d.arr" % i
            comp_name = "original_%d.arr.zfp" % i
            gzip_name = comp_name + ".gz"
            round_trip_name = "round_trip_%d.arr" % i            
            
            tile.write_to_file(orig_name, "f4")
            
            subprocess.run((self.zfp, "-i", orig_name, "-2", str(self.tile_size), str(self.tile_size), "-f", *args, "-z", comp_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE)            
            subprocess.run((self.zfp, "-z", comp_name, "-2", str(self.tile_size), str(self.tile_size), "-f", *args ,"-o", round_trip_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(("gzip", "-1", comp_name))
            compressed_size += os.stat(gzip_name).st_size
            round_trip_tiles.append(Region.from_file(round_trip_name, "f4", self.tile_size, self.tile_size))
            
            subprocess.run(("rm", orig_name))
            subprocess.run(("rm", gzip_name))
            subprocess.run(("rm", round_trip_name))
            
        round_trip_region = Region.from_tiles(round_trip_tiles, self.region.data.shape)
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
        
        round_trip_tiles = []
        compressed_size = 0
                
        for i, tile in enumerate(self.region.tiles(self.tile_size)):
            orig_name = "original_%d.arr" % i
            comp_name = "original_%d.arr.sz" % i
            gzip_name = comp_name + ".gz"
            round_trip_name = "original_%d.arr.sz.out" % i
            
            tile.write_to_file(orig_name, "f4")
            
            config_path = os.path.join(prev, "sz.config")
            subprocess.run((self.sz, "-c", config_path, *args, "-f", "-z", "-i", orig_name, "-2", str(self.tile_size), str(self.tile_size)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run((self.sz, "-c", config_path, *args, "-f", "-x", "-s", comp_name, "-2", str(self.tile_size), str(self.tile_size)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(("gzip", "-1", comp_name))
            compressed_size += os.stat(gzip_name).st_size
            round_trip_tiles.append(Region.from_file(round_trip_name, "f4", self.tile_size, self.tile_size))

            subprocess.run(("rm", orig_name))
            subprocess.run(("rm", gzip_name))
            subprocess.run(("rm", round_trip_name))
        
        round_trip_region = Region.from_tiles(round_trip_tiles, self.region.data.shape)
        compressed_image = self.image.clone_colourmap_to(round_trip_region)
        
        os.chdir(prev)
        return round_trip_region, compressed_image, compressed_size, None
        
    def SZ_compress_PSNR(self, PSNR):
        return self.SZ_compress("-M", "PSNR", "-S", str(PSNR))

    def JPG_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        compressed_tiles = []
        compressed_size = 0
        
        for i, tile in enumerate(self.image.tiles(self.tile_size)):
            comp_name = "compressed_%d.jpg" % i
        
            tile.to_jpg(comp_name, args[0])
        
            compressed_size += os.stat(comp_name).st_size
            compressed_tiles.append(ColourmappedRegion.from_jpg(comp_name))

            subprocess.run(("rm", comp_name))
            
        compressed_image = ColourmappedRegion.from_tiles(compressed_tiles, self.image.image.shape)
        
        os.chdir(prev)
        return None, compressed_image, None, compressed_size

    def JPG_compress_quality(self, quality):
        return self.JPG_compress(quality)
    
    def BPG_compress(self, *args):
        prev = os.getcwd()
        os.chdir(self.temp_dir)
        
        compressed_tiles = []
        compressed_size = 0
        
        for i, tile in enumerate(self.image.tiles(self.tile_size)):
            orig_name = "uncompressed_%d.png" % i
            comp_name = "compressed_%d.bpg" % i
            round_trip_name = "round_trip_%d.png" % i
        
            tile.to_png(orig_name)
            subprocess.run((self.bpgenc, *args, orig_name, "-o", comp_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            compressed_size += os.stat(comp_name).st_size
            
            subprocess.run((self.bpgdec, comp_name, "-o", round_trip_name), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            compressed_tiles.append(ColourmappedRegion.from_png(round_trip_name))
            
            subprocess.run(("rm", orig_name))
            subprocess.run(("rm", comp_name))
            subprocess.run(("rm", round_trip_name))
        
        compressed_image = ColourmappedRegion.from_tiles(compressed_tiles, self.image.image.shape)
        
        os.chdir(prev)
        return None, compressed_image, None, compressed_size
    
    def BPG_compress_quantiser(self, quantiser):
        return self.BPG_compress("-q", str(quantiser))


class Comparator:
    def __init__(self, image, results, compressor):
        self.image = image
        self.results = results
        self.compressor = compressor
    
    ALGORITHMS = {
        #"ZFP (Fixed rate)": ["ZFP_compress_fixed_rate", range(1, 24)],
        "ZFP (Fixed precision)": ["ZFP_compress_fixed_precision", range(4, 28)],
        #"ZFP (Fixed accuracy)": [
        #        "ZFP_compress_fixed_accuracy",
        #        list(range(1, 21)) + [
        #            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        #            5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 
        #            5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7
        #        ]
        #    ],
        "SZ (PSNR bounded)": ["SZ_compress_PSNR", range(40, 110, 5)],
        "JPEG": ["JPG_compress_quality", range(60, 101, 2)],
        "BPG (quantizer)": ["BPG_compress_quantiser", range(52)],
    }
    
    # Uncomment for testing during development
    #ALGORITHMS = {
        #"ZFP (Fixed rate)": ["ZFP_compress_fixed_rate", range(1, 32+1, 10)],
        #"JPEG": ["JPG_compress_quality", range(60, 101, 10)],
    #}
        
    PLOT_COLOURS = {
        "ZFP (Fixed rate)": "red",
        "ZFP (Fixed precision)": "orange",
        "ZFP (Fixed accuracy)": "yellow",
        "SZ (PSNR bounded)": "green",
        "JPEG": "blue",
        "BPG (quantizer)": "magenta",
    }
    
    ERROR_FUNCTION_NAMES = ("mean", "max", "median")

    @classmethod
    def compare_algorithms(cls, region, colourmap, temp_dir=".", zfp="zfp", sz="sz", bpgenc="bpgenc", bpgdec="bpgdec", logarithmic=False, nan_interpolation_method=None, bpg_quant_step=4, tile_size=256):
        for d in region.data.shape:
            if d % tile_size:
                raise ValueError("Image dimension %d is not divisible by tile size %d. Aborting." % (d, tile_size))
        
        results = []
        
        original_raw_size = np.dtype("f4").itemsize * region.data.size
        original_region = region
        
        if nan_interpolation_method is not None:
            region = Region(fix_nans(region.data, nan_interpolation_method))
        
        image = region.colourmapped(colourmap, log=logarithmic)
        image.to_png("original.png")
        original_image_size = os.stat("original.png").st_size
        
        compressor = Compressor(region, image, temp_dir, zfp, sz, bpgenc, bpgdec)
        
        if bpg_quant_step > 1 and "BPG (quantizer)" in cls.ALGORITHMS:
            cls.ALGORITHMS["BPG (quantizer)"][1] = range(0, 52, bpg_quant_step)
                
        for label, (function_name, params) in cls.ALGORITHMS.items():
            
            for p in params:
                result_dict = {
                    "label": label,
                    "function_name": function_name, # may need it later to regenerate images
                    "param": p, # may need it later to regenerate images}
                }
                
                print("Running algorithm %s with parameter %d..." % (label, p))
                
                round_trip_region, compressed_image, compressed_raw_size, compressed_image_size = getattr(compressor, function_name)(p)
                
                # ERRORS: raw data and image; absolute and relative; multiple functions
                
                error_functions = [getattr(np, "nan" + n) for n in cls.ERROR_FUNCTION_NAMES]
                
                if round_trip_region:
                    # these are (absolute, relative) pairs
                    # We compare the raw data to the original raw data *before* nans are interpolated
                    raw_errors = original_region.delta_errors(round_trip_region, *error_functions)
                else:
                    raw_errors = [(None, None)] * len(error_functions)
                
                image_errors = image.delta_errors(compressed_image, *error_functions, mask=~np.isnan(original_region.data))
                
                for func_name, (raw_abs, raw_rel), (img_abs, img_rel) in zip(cls.ERROR_FUNCTION_NAMES, raw_errors, image_errors):
                    result_dict["raw_error_%s_abs" % func_name] = raw_abs
                    result_dict["raw_error_%s_rel" % func_name] = raw_rel
                    result_dict["image_error_%s_abs" % func_name] = img_abs
                    result_dict["image_error_%s_rel" % func_name] = img_rel
                
                if compressed_raw_size:
                    result_dict["size_fraction"] = compressed_raw_size/original_raw_size
                elif compressed_image_size:
                    result_dict["size_fraction"] = compressed_image_size/original_raw_size
                
                results.append(result_dict)

                    
        subprocess.run(("rm", "original.png"))
        
        return cls(image, results, compressor)

    def get(self, fields, where):
        return [[r[f] for f in fields] for r in self.results if all(comp(r[k], v) for k, (v, comp) in where.items())]
    
    def unique(self, field):
        return {r[field] for r in self.results}
        
    def plot(self, xfield, yfield, xlabel, ylabel, plt_obj, xmin, xmax, ymin, ymax):
        for label in self.unique("label"):
            xy = sorted(self.get((xfield, yfield), {"label": (label, operator.eq)})) # sort by xfield
            
            if all(y is None for x, y in xy):
                continue # skip e.g. non-existent raw errors for JPEG
                        
            if xmin is not None:
                xy = [(x, y) for x, y in xy if x >= xmin]
            if xmax is not None:
                xy = [(x, y) for x, y in xy if x <= xmax]
            if ymin is not None:
                xy = [(x, y) for x, y in xy if y >= ymin]
            if ymax is not None:
                xy = [(x, y) for x, y in xy if y <= ymax]
                        
            if not xy:
                continue
            
            x, y = zip(*xy)
            
            plt_obj.plot(x, y, marker='o', ls='', label=label, color=self.PLOT_COLOURS[label])
            
        plt_obj.legend()
        
        if hasattr(plt_obj, "set_xlabel"):
            plt_obj.set_xlabel(xlabel)
            plt_obj.set_ylabel(ylabel)
        else:
            plt_obj.xlabel(xlabel)
            plt_obj.ylabel(ylabel)
    
    def show_plots(self, plots, datasets, error_type, xmin, xmax, ymin, ymax, width, height):
        plt.rcParams['figure.figsize'] = (width, height)
        
        nrows = len(plots)
        ncols = len(datasets)
                        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        
        def float_or_none(val):
            try:
                return float(val)
            except ValueError:
                return None
        
        xmin = float_or_none(xmin)
        xmax = float_or_none(xmax)
        ymin = float_or_none(ymin)
        ymax = float_or_none(ymax)
        
        for i, plot in enumerate(plots):
            for j, dataset in enumerate(datasets):
                self.plot("size_fraction", "%s_error_%s_%s" % (dataset, plot, error_type[:3]), "Size (fraction)", 
                        "%s error (%s %s)" % (dataset, error_type, plot), axs[i][j], xmin, xmax, ymin, ymax)
                
    def widget_plots(self):
        return interact(
            self.show_plots, 
            plots=SelectMultiple(options=self.ERROR_FUNCTION_NAMES, value=["mean", "max"], description="Plots"), 
            datasets=SelectMultiple(options=["raw", "image"], value=["image"], description="Datasets"),
            error_type=Dropdown(options={"Absolute": "absolute", "Relative": "relative"}, value="absolute", description='Error'),
            xmin=Text(value="", placeholder="Type a number or leave blank to disable", description='Min x'),
            xmax=Text(value="", placeholder="Type a number or leave blank to disable", description='Max x'),
            ymin=Text(value="", placeholder="Type a number or leave blank to disable", description='Min y'),
            ymax=Text(value="", placeholder="Type a number or leave blank to disable", description='Max y'),
            width=IntSlider(value=15, min=5, max=50, step=1, continuous_update=False, description="Subplot width"), 
            height=IntSlider(value=15, min=5, max=50, step=1, continuous_update=False, description="Subplot height")
        )

    def show_images(self, size, show, algorithm):
        label = algorithm
        
        if not algorithm.startswith("None"):
            results = sorted(self.get(("size_fraction", "function_name", "param", "image_error_mean_abs", "image_error_mean_rel"), {"label": (label, operator.eq), "size_fraction": (size, operator.le)}))
            
            selected_image = None
            
            if results:
                size_fraction, function_name, p, error_a, error_r = results[-1]
                round_trip_region, compressed_image, compressed_raw_size, compressed_image_size = getattr(self.compressor, function_name)(p)
                print("%s with parameter %d: size %.2f, error %.2g (absolute) %1.2e (relative)" % (label, p, size_fraction, error_a, error_r))
                selected_image = compressed_image
        
        empty = Image.fromarray(np.zeros(self.image.image.shape).astype(np.uint8))
        full = Image.fromarray((np.ones(self.image.image.shape) * 255).astype(np.uint8))
        
        if algorithm.startswith("None"): # exact image
            if show == "image":
                self.image.display()
            else: # we're showing the difference
                empty.show()
                
        elif selected_image is None: # No image found for this algorithm
            if show == "image":
                empty.show()
            else: # we're showing the difference
                full.show()
                
        else: # image found
            if show == "image":
                selected_image.display()
            else:
                diff = self.image.image - selected_image.image + 0.5
                Image.fromarray((diff * 255).astype(np.uint8)).show()
                        
    def widget_images(self):
        return interact(
            self.show_images, 
            size=FloatSlider(value=0.5, min=0, max=1, step=0.01, continuous_update=False, description="Size fraction"),
            show=Select(options={"Image": "image", "Difference": "difference"}, value="image", description='Show'),
            algorithm=Select(options=['None (exact)'] + list(self.ALGORITHMS.keys()), value='JPEG', description='Algorithm')
        )
