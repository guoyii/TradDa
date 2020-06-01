import numpy as np 
import copy
import sys 


## Inter Function
def SinoInter(sinogram_LineInter, geo_full, weg=1, option = "sinogram_LineInter"):
    # sinogram_exten = my_extension(sinogram_sparse, (geo_full["sino_views"], geo_full["nDetecU"]))
    """
    geo = {"nVoxelX": image_size, "nVoxelY": image_size, 
       "sVoxelX": image_size, "sVoxelY": image_size, 
       "dVoxelX": 1.0, "dVoxelY": 1.0, 
       "sino_views": views, 
       "nDetecU": 736, "sDetecU": 736.0,
       "dDetecU": 1.0, "DSD": 600.0, "DSO": 550.0, "DOD": 50.0,
       "offOriginX": 0.0, "offOriginY": 0.0, 
       "offDetecU": 0.0,
       "start_angle": 0, "end_angle": np.pi,
       "accuracy": 0.5, "mode": "parallel", 
       "extent": 3,
       "COR": 0.0}
    return geo
    """
    angles = geo_full["end_angle"] - geo_full["start_angle"]
    angle = angles/geo_full["sino_views"]
    deta_length = geo_full["DSD"] * np.sin(angle)
    # print("deta_length:",deta_length)

    # sinogram_inter = np.zeros((geo_full["sino_views"], geo_full["nDetecU"]),dtype=np.float64)
    # view_index = (np.linspace(0, geo_full["sino_views"]-1, sinogram_sparse.shape[0])).astype(np.int32)
    # for i,index in enumerate(view_index):
    #     sinogram_inter[index] = sinogram_sparse[i]
    
    sinogram_inter_z = copy.copy(sinogram_LineInter)
    sinogram_inter_f = copy.copy(sinogram_LineInter)
    for i in range(geo_full["sino_views"]):
        if i==0:
            pass
        else:
            for index in range(geo_full["nDetecU"]):
                y = geo_full["nDetecU"]/2 - index
                temp = y/np.tan(angle)
                left_length = geo_full["DOD"] + temp
                start_index = geo_full["nDetecU"]/2 - left_length * np.sin(angle)
                end_index = start_index + deta_length
                end_index = int(end_index)
                start_index = int(start_index)
                if start_index < 0:
                    start_index = 0
                if end_index > geo_full["nDetecU"]-1:
                    end_index = geo_full["nDetecU"]-1 
                if option is "sinogram_LineInter":
                    avg = sinogram_LineInter[i-1, start_index:end_index+1].sum()
                else:
                    avg = sinogram_inter_z[i-1, start_index:end_index+1].sum()
                fenmu = end_index - start_index+2
                sinogram_inter_z[i, index] = (avg*(fenmu-weg)/(fenmu-1) + weg*sinogram_inter_z[i, index])/fenmu
                
    for i in range(geo_full["sino_views"]):
        i = geo_full["sino_views"]-1-i
        if i==geo_full["sino_views"]-1:
            pass
        else:
            for index in range(geo_full["nDetecU"]):
                y = geo_full["nDetecU"]/2 - index
                temp = geo_full["DOD"] * np.tan(angle/2)
                end_index = geo_full["nDetecU"]/2 - ((y-temp)*np.cos(angle)-temp)
                start_index = int(end_index - deta_length)
                end_index = int(end_index)
                if start_index < 0:
                    start_index = 0
                if end_index > geo_full["nDetecU"]-1:
                    end_index = geo_full["nDetecU"]-1 
                if option is "sinogram_LineInter":
                    avg = sinogram_LineInter[i+1, start_index:end_index+1].sum()
                else:
                    avg = sinogram_inter_f[i+1, start_index:end_index+1].sum()
                fenmu = end_index - start_index+2
                sinogram_inter_f[i, index] = (avg*(fenmu-weg)/(fenmu-1) + weg*sinogram_inter_f[i, index])/fenmu
    return (sinogram_inter_z + sinogram_inter_f)/2
    # return sinogram_inter_f