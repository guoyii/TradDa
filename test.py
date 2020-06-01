import numpy as np 
from torch.utils.data import Dataset
from idea_function import my_extension
from datasets_function import my_map_coordinates
import math 
def ass(a):
    a = a+1
    b = a
    return b
# a = np.random.rand(4,5)
a = np.array([[1, 2, 3, 4, 5],
              [11, 12, 13, 14, 15],
              [21, 22, 23, 24, 25],
              [31, 32, 33, 34, 35]])
b = my_map_coordinates(a, (10,5), 1)
print(a)
print(b)
