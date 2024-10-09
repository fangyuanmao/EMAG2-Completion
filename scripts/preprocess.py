import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
img_write_q = [int(cv2.IMWRITE_PNG_COMPRESSION),0]

# Calculate masks.
def mask(array):
    array[np.where(array < 99999)] = 1
    array[np.where(array == 99999)] = 0
    return array

# Normalization.
def Normalize(array, mx=200, mn=-200):
    # Normalize and filter.
    array[np.where(array > mx)] = mx
    array[np.where(array < mn)] = mn
    array = (array-mn)/(mx-mn)
    return array

# Select the anomaly values only.
data = np.loadtxt("../EMAG2/EMAG2_V3.csv", skiprows=0, delimiter=',', usecols=(5))
data = data.reshape(5400,10800)
data = np.flipud(data)
np.savetxt('./EMAG2/EMAG2_V3_values.csv',data,delimiter=', ', fmt='%f')

# Slice the EMAG2 data into 10*20.
PATH = './EMAG2/EMAG2_split_csv_200/'
os.makedirs(PATH, exist_ok=True)
split_lat = np.split(data, 10, axis=0)
for latidx, splitlat in tqdm(enumerate(split_lat)):
    split_lat_lon = np.split(splitlat, 20, axis=1)
    for lonidx, splitlatlon in enumerate(split_lat_lon):
        splitlatlon = np.flipud(splitlatlon)
        np.savetxt(PATH+'lat'+str(9-latidx)+'-'+'lon'+str(lonidx)+'.csv',splitlatlon,delimiter=', ', fmt='%f')

# PATH.
INPUTPATH = './EMAG2/EMAG2_split_csv_200/'
OUTPUTPATH = './EMAG2/EMAG2_split_png_200/'
OUTPUTMASKPATH = './EMAG2/EMAG2_split_mask_200/'
os.makedirs(OUTPUTPATH, exist_ok=True)
os.makedirs(OUTPUTMASKPATH, exist_ok=True)
filenames = os.listdir(INPUTPATH)

# Calculate masks.
for file in tqdm(filenames):
    data = np.loadtxt(os.path.join(INPUTPATH, file), skiprows=0, delimiter=',')
    image_array = mask(data)
    image_array *= 255 
    cv2.imwrite(os.path.join(OUTPUTMASKPATH,file.replace('csv','png')), image_array, img_write_q)
   
# Convert csv to image.
for file in tqdm(filenames):
    data = np.loadtxt(os.path.join(INPUTPATH, file), skiprows=0, delimiter=',')
    image_array = Normalize(data, mx=200, mn=-200)
    image_array *= 255 
    cv2.imwrite(os.path.join(OUTPUTPATH,file.replace('csv','png')), image_array, img_write_q)
