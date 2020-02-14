# %%
import cv2
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import os, shutil
import random
from skimage import util, data, filters, segmentation, measure, morphology, color


# set directory parameters
original_dataset_dir = '../../dataset/original/'

# get file count
img_list = os.listdir(original_dataset_dir)
print(len(img_list))



# %%
# file name parser
def parse_file_name(img_path):
    finger_id, _ = os.path.splitext(os.path.basename(img_path))
    #print(finger_id)
    return np.array([finger_id], dtype=np.uint16)

def parse_file_name2(img_path):
    finger_id, _ = os.path.splitext(os.path.basename(img_path))
    finger_id, finger_index = finger_id.split('_')
    #print(finger_id, finger_index)
    return np.array([finger_id], dtype=np.uint16)



# %%
# convert tif to bmp
finger_idx = 0
finger_id = -1
file_id = 0
for item in img_list:
    if item == '.DS_Store':
        continue
    if item == 'README.md':
        continue

    finger_id_tmp = parse_file_name2(item)
    if (int(finger_id_tmp) != file_id):
        file_id = int(finger_id_tmp)
        finger_id += 1
        finger_idx = 0

    img_path = original_dataset_dir + item

    # open file
    img = Image.open(img_path)

    # save file
    img_new_path = original_dataset_dir + '{0:05d}'.format(finger_id) + '_' + '{0:02d}'.format(finger_idx) + '.bmp'
    img.save(img_new_path)
    finger_idx += 1

print('Complete')


# %%
