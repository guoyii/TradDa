import numpy as np
import pydicom
import torch
import math 
import glob
import astra
from scipy.ndimage import map_coordinates


## Cut image randomly
##***********************************************************************************************************
class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        image = np.hstack((image, image))
        image = np.vstack((image, image))

        crop_point = np.random.randint(self.crop_size, size=2)
        image = image[crop_point[0]:crop_point[0]+self.crop_size, crop_point[1]:crop_point[1]+self.crop_size]
        image = np.pad(image,((math.ceil((self.crop_size - image.shape[0])/2), math.floor((self.crop_size - image.shape[0])/2)),
                (math.ceil((self.crop_size - image.shape[1])/2), math.floor((self.crop_size - image.shape[1])/2))),"constant")
        return image
        

## Extense sparse sinogram with zeros to the size of full views sinogram
##***********************************************************************************************************
def my_extension(sinogram_sparse, size):
    sinogram_inter = np.zeros(size)
    view_index = (np.linspace(0, size[0]-1, sinogram_sparse.shape[0])).astype(np.int32)
    for i,index in enumerate(view_index):
        sinogram_inter[index] = sinogram_sparse[i]
    return sinogram_inter

 
## Read Mayo Image
##***********************************************************************************************************
def findFiles(path): return glob.glob(path)


## Read Mayo Image
##***********************************************************************************************************
def image_read(image_path):
    return pydicom.dcmread(image_path)


## Normalize
##***********************************************************************************************************
class Normalize(object):
    def __init__(self, normalize_type="self"):
        if normalize_type is "image":
            self.mean = 128.0
        elif normalize_type is "sino":
            self.mean = -150.0
        elif normalize_type is "self":
            self.mean = None

    def __call__(self, image):
        if self.mean is not None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)

        img_var = np.var(image)

        image = image - img_mean
        image = image / img_var

        return image


## Image normalization
##***********************************************************************************************************
class Any2One(object):
    def __call__(self, image):
        image_max = np.max(image)
        image_min = np.min(image)
        return (image-image_min)/(image_max-image_min)




## Change to torch
##***********************************************************************************************************
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        return torch.from_numpy(image).type(torch.FloatTensor)

## Add noise
##***********************************************************************************************************
def add_noise(noise_typ, image, mean=0, var=0.1):
    if noise_typ == "gauss":
        row,col= image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        noisy = np.random.poisson(image)
        return noisy


## sparse transform
##***********************************************************************************************************
def sparse_view_f(sino_true,  view_origin=1160, view_sparse=60):
   view_index = (np.linspace(0, view_origin-1, view_sparse)).astype(np.int32)
   return sino_true[view_index, :]


## resize image
##***********************************************************************************************************
def my_map_coordinates(image, size, order = 1):
    ## scipy.ndimage.map_coordinates 
   new_dims = []
   for original_size, new_size in zip(image.shape, size):
      new_dims.append(np.linspace(0, original_size-1, new_size))
   coords = np.meshgrid(*new_dims, indexing='ij')
   """
   Parameters:	
   input : ndarray
   The input array.
   coordinates : array_like
   The coordinates at which input is evaluated.
   output : ndarray or dtype, optional
   The array in which to place the output, or the dtype of the returned array.
   order : int, optional
   The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
   mode : str, optional
   Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
   cval : scalar, optional
   Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
   prefilter : bool, optional
   The parameter prefilter determines if the input is pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it is assumed that the input is already filtered. Default is True.
   Returns:	
   map_coordinates : ndarray
   The result of transforming the input. The shape of the output is derived from that of coordinates by dropping the first axis.
   """
   return map_coordinates(image, coords, order=order)



