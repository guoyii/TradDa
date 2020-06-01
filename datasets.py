import torch
import astra
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from datasets_function import RandomCrop, Normalize, Any2One, ToTensor
from datasets_function import findFiles, image_read
from datasets_function import my_extension, my_map_coordinates, sparse_view_f
import copy
import os

## Basic datasets
##***********************************************************************************************************
class BasicData(Dataset):
    def __init__(self, data_root_path, folder, crop_size=None, Dataset_name="test"):
        self.Dataset_name = Dataset_name
        self.crop_size = crop_size
        self.fix_list = [RandomCrop(self.crop_size), Normalize(), Any2One(), ToTensor()]

        if Dataset_name is "train":
            self.image_paths = [findFiles(data_root_path + "/{}/{}/*.IMA".format(x, y)) for x in folder["patients"] for y in folder["SliceThickness"]]
            self.image_paths = [x for j in self.image_paths for x in j]
        else:
            self.image_paths = findFiles("{}/{}/{}/*.IMA".format(data_root_path, folder["patients"], folder["SliceThickness"]))
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = image_read(image_path)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept
        imgname = os.path.splitext(os.path.split(image_path)[1])[0]

        transform = transforms.Compose(self.fix_list)
        imgdata = transform(imgdata).numpy()  

        return imgdata, imgname


class BuildDataSet(Dataset):
    def __init__(self, data_root_path, folder, geo_full, geo_sparse, Dataset_name="train"):
        self.Dataset_name = Dataset_name
        self.geo_full = geo_full
        self.geo_sparse = geo_sparse
        self.imgset = BasicData(data_root_path, folder, self.geo_full["nVoxelX"], Dataset_name)
        ## Full-----------------------------------------
        self.vol_geom = astra.create_vol_geom(self.geo_full["nVoxelY"], self.geo_full["nVoxelX"], 
                                            -1*self.geo_full["sVoxelY"]/2, self.geo_full["sVoxelY"]/2, -1*self.geo_full["sVoxelX"]/2, self.geo_full["sVoxelX"]/2)
        self.proj_geom = astra.create_proj_geom(self.geo_full["mode"], self.geo_full["dDetecU"], self.geo_full["nDetecU"], 
                                                np.linspace(self.geo_full["start_angle"], self.geo_full["end_angle"], self.geo_full["sino_views"],False), self.geo_full["DSO"], self.geo_full["DOD"])
        if self.geo_full["mode"] is "parallel":
            self.proj_id = astra.create_projector("linear", self.proj_geom, self.vol_geom)
        elif self.geo_full["mode"] is "fanflat":
            self.proj_id = astra.create_projector("line_fanflat", self.proj_geom, self.vol_geom)

    @classmethod
    def project(cls, image, proj_id):
        sinogram_id, sino = astra.create_sino(image, proj_id) 
        astra.data2d.delete(sinogram_id)
        sinogram = copy.deepcopy(sino)
        return sinogram 

    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        image, image_name = self.imgset[idx]
        
        sinogram_full = self.project(image, self.proj_id)
        sinogram_sparse = sparse_view_f(sinogram_full, self.geo_full["sino_views"], self.geo_sparse["sino_views"])
        sinogram_inter = my_map_coordinates(sinogram_sparse, (self.geo_full["sino_views"], self.geo_full["nDetecU"]), order=1)
        
        return image, sinogram_full, sinogram_sparse, sinogram_inter, image_name

