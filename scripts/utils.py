import csv
import numpy as np
from scipy import ndimage
import os
import shutil
from PIL import Image
from copy import deepcopy
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm

def read_csv(csv_filename='./EMAG2/EMAG2_V3_values.csv'):
    data = np.loadtxt(csv_filename, delimiter=',', skiprows=0)
    return data


def resize_and_convert(img, size, resample=Image.BICUBIC):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

def shift_center(center, h, w, size=512):
    y = center[0]
    x = center[1]
    if y - size//2 < 0:
        y = size//2
    elif y + size//2 > h-1:
        y = h - size//2
    else:
        pass
    
    if x - size//2 < 0:
        x = size//2
    elif x + size//2 > w-1:
        x = w - size//2
    else:
        pass
    return y,x

def Normalize(array, mx=200, mn=-200):
    # Normalize and filter
    array[np.where(array > mx)] = mx
    array[np.where(array < mn)] = mn
    array = (array-mn)/(mx-mn)
    return array

def mask_filter(img):
    img = np.array(img)
    return np.where(img < 255, 0, 255)

def graystyle(img):
    return img.convert('L')

def save_img(image, output_path, name, resize=512, mask_fig=False):
    img = Image.fromarray(image)
    if resize is not None:
        img = resize_and_convert(img, resize)
    if mask_fig:
        img = mask_filter(img)
        img = Image.fromarray(np.uint8(img))
    img = graystyle(img)
    img.save(os.path.join(output_path, f'example_{name}.png'))
    
def paint(image, path, name, Normalization=None, resize=512, mask_fig=False):
    if Normalization is not None:
        image = Normalize(image, mn=Normalization[0], mx=Normalization[1])
    else:
        pass
    image *= 255
    save_img(image, path, name, resize=resize, mask_fig=mask_fig)
    
def data_slice(new_y, new_x, data, data_anomaly_mask, labeled_mask, label, length_of_side):
    repaint_img  = deepcopy(data[new_y-length_of_side//2:new_y+length_of_side//2, 
                                 new_x-length_of_side//2:new_x+length_of_side//2])
    repaint_mask = deepcopy(data_anomaly_mask[new_y-length_of_side//2:new_y+length_of_side//2, 
                                              new_x-length_of_side//2:new_x+length_of_side//2])
    repaint_mask = np.where(repaint_mask > 0.5, 0.0, 1.0)
    recover_mask = deepcopy(labeled_mask[new_y-length_of_side//2:new_y+length_of_side//2, 
                                         new_x-length_of_side//2:new_x+length_of_side//2])
    recover_mask = np.where(recover_mask == label, 0.0, 1.0)
    return repaint_img, repaint_mask, recover_mask
            
def gray2mag(a):
    return 400 * ( a / 255) - 200

def create_mask(data, exception_value=99999):
    mask = np.zeros_like(data, dtype=bool)
    exception_positions = np.where(data == exception_value)
    mask[exception_positions] = True
    return mask

def centers_and_labeled_mask(mask, mask_size):
    centers = []
    labeled_mask, num_labels = ndimage.label(mask)
    
    with open('./EMAG2/centers_all.txt', 'r+') as f:
        with open('./EMAG2/centers_mask_'+str(mask_size)+'.txt', 'w+') as fw:
            for i in tqdm(f.readlines()):
                indices = i.strip().split(',')            
                if int(indices[2]) > mask_size // 2:
                    centers.append((int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])))
                    fw.write(indices[0]+','+indices[1]+','+indices[2]+','+indices[3]+'\n')
            
    # # calculate certer of masks, but it consumes lots of time! Please use the center_all.txt.
    #     for label in tqdm(range(1, num_labels + 1)):
    #         labeled_indices = np.where(labeled_mask == label)
    #         min_row = np.min(labeled_indices[0])
    #         max_row = np.max(labeled_indices[0])
    #         min_col = np.min(labeled_indices[1])
    #         max_col = np.max(labeled_indices[1])
    #         center_y = int(np.mean(labeled_indices[0]))
    #         center_x = int(np.mean(labeled_indices[1]))
    #         length_of_side = max([center_y-min_row, max_row-center_y, center_x-min_col, max_col-center_x])
    #         with open('center.txt', 'a+') as f:
    #             f.write(center_y+','+center_x+','+length_of_side+','+label+'\n')
             
    return centers, labeled_mask

def generate_img_mask(data, data_anomaly_mask, centers, labeled_mask, output_path='./debug', size=512):
    os.makedirs(output_path, exist_ok=True)
    shutil.rmtree(output_path)
    img_path = os.path.join(output_path, 'img')
    repaint_mask_path = os.path.join(output_path, 'repaint_mask')
    recover_mask_path = os.path.join(output_path, 'recover_mask')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(repaint_mask_path, exist_ok=True)
    os.makedirs(recover_mask_path, exist_ok=True)
    h, w = data.shape
    
    with open(os.path.join(output_path, 'centers.txt'), 'a+') as f:
        for i, center in tqdm(enumerate(centers)):
            label = center[3]
            length_of_side = max(center[2]*2, size)
            
            # shift center to avoid overflow
            new_y, new_x = shift_center(center, size=length_of_side, h=h, w=w)
            
            # slice
            repaint_img, repaint_mask, recover_mask = data_slice(
                new_y, new_x, data, data_anomaly_mask, labeled_mask, label, length_of_side
                )
            
            # paint
            paint(repaint_img, img_path, label, Normalization=(-200,200), resize=size)
            paint(repaint_mask, repaint_mask_path, label, resize=size, mask_fig=True)
            paint(recover_mask, recover_mask_path, label, resize=size, mask_fig=True)
            f.write(str(new_y) + ',' + str(new_x) + ',' + str(center[2]) + ','  + str(label) +'\n')

def merge_centers(args):
    centers=[]
    if args.local:
        with open(os.path.join(args.inpainted_path, 'centers.txt'), 'r+') as f:
            for line in tqdm(f.readlines()):
                indices = line.strip().split(',')
                centers.append((int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])))
    else:
        for lat in tqdm(range(0, 10)):
            for lon in range(0, 20):
                centers.append((lat*args.size+args.size//2, lon*args.size+args.size//2, args.size//2, -1))
    return centers

def data_mix(emag2_ori, centers, args):
    completed_data = deepcopy(emag2_ori)
    img_path = os.path.join(args.inpainted_path, 'img')
    mask_path = os.path.join(args.inpainted_path, 'mask')
    
    if args.local:
        imgfiles  = os.listdir(img_path)
        maskfiles = os.listdir(img_path)
        for center in tqdm(centers):
            y = center[0]
            x = center[1]
            length_of_side = max(center[2], 256)
            label = center[3]
            slice = f'example_{label}.png'
            inpainted_img  = graystyle(Image.open(os.path.join(img_path, slice)))
            inpainted_mask = graystyle(Image.open(os.path.join(mask_path, slice)))
            inpainted_img  = np.array(resize_and_convert(inpainted_img, length_of_side*2))
            inpainted_mask = np.array(resize_and_convert(inpainted_mask, length_of_side*2))
            inpainted_img  = gray2mag(inpainted_img)
            inpainted_mask = mask_filter(inpainted_mask)
            completed_data[y-length_of_side:y+length_of_side, x-length_of_side:x+length_of_side][np.where(inpainted_mask==0)]=inpainted_img[np.where(inpainted_mask==0)]
    else:
        for lat in tqdm(range(0, 10)):
            for lon in range(0, 20):
                slice = f'lat{lat}-lon{lon}.png'
                inpainted_img  = graystyle(Image.open(os.path.join(img_path, slice)))
                inpainted_mask = graystyle(Image.open(os.path.join(mask_path, slice)))
                inpainted_img  = np.array(resize_and_convert(inpainted_img, args.size))
                inpainted_mask = np.array(resize_and_convert(inpainted_mask, args.size))
                inpainted_img  = gray2mag(inpainted_img)
                inpainted_mask = mask_filter(inpainted_mask)
                completed_data[lat*args.size:(lat+1)*args.size, lon*args.size:(lon+1)*args.size][np.where(inpainted_mask==0)] = inpainted_img[np.where(inpainted_mask==0)]
    return completed_data
            
def merge_emag2(emag2_ori, args=None):
    os.makedirs(args.output_path, exist_ok=True)
    centers = merge_centers(args)
    return data_mix(emag2_ori, centers, args)
   

            
            
            
            
            
            
            
            
            
            