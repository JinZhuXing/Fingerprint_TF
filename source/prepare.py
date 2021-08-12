from PIL import Image, ImageOps
import numpy as np
import os
import random


# global variables
EXPAND_WIDTH = 230
EXPAND_HEIGHT = 230


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

# Otsu Binalization
def otsu_binarization(img, th = 128):
    H, W = img.shape
    out = img.copy()

    max_sigma = 0
    max_t = 0

    # determine threshold
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    # print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255

    return out


# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    out[y, x] = 255

    return out


# Morphology Erode
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y, x] = 0

    return out


# get fingerprint region for crop
def get_fp_region(img_path, crop_width, crop_height):
    CropWidth = crop_width
    CropHeight = crop_height
    ExpWidth = EXPAND_WIDTH
    ExpHeight = EXPAND_HEIGHT

    image = Image.open(img_path)
    image_buf = np.asarray(image)

    # otsu binarization
    otsu = otsu_binarization(image_buf)

    # morphology
    cleared = Morphology_Erode(otsu, Erode_time = 2)

    img_width = image_buf.shape[1]
    img_height = image_buf.shape[0]
    #print(img_width, img_height)

    crop_l = img_width
    crop_r = 0
    crop_t = img_height
    crop_b = 0
    for i in range(img_height):
        for j in range(img_width):
            if cleared[i, j] == 0:
                if (crop_l > j):
                    crop_l = j
                if (crop_r < j):
                    crop_r = j
                if (crop_t > i):
                    crop_t = i
                if (crop_b < i):
                    crop_b = i

    if ((crop_r - crop_l) < CropWidth):
        diff = CropWidth - (crop_r - crop_l)
        if (crop_r + crop_l > CropWidth): # right
            if (img_width - crop_r > diff / 2):
                crop_r += diff / 2
                crop_l -= diff / 2
            else:
                crop_r = img_width - 1
                crop_l = crop_r - (CropWidth + 2)
        else: # left
            if (crop_l > diff / 2):
                crop_l -= diff / 2
                crop_r += diff / 2
            else:
                crop_l = 1
                crop_r = crop_l + (CropWidth + 2)
    if ((crop_b - crop_t) < CropHeight):
        diff = CropHeight - (crop_b - crop_t)
        if (crop_b + crop_t > CropHeight): # bottom
            if (img_height - crop_b > diff / 2):
                crop_b += diff / 2
                crop_t -= diff / 2
            else:
                crop_b = img_height - 1
                crop_t = crop_b - (CropHeight + 2)
        else: # top
            if (crop_t > diff / 2):
                crop_t -= diff / 2
                crop_b += diff / 2
            else:
                crop_t = 1
                crop_b = crop_t + (CropHeight + 2)

    # expand region for rotation
    crop_l = (crop_r + crop_l - CropWidth) / 2
    crop_r = crop_l + CropWidth
    crop_t = (crop_t + crop_b - CropHeight) / 2
    crop_b = crop_t + CropHeight
    crop_l = (int)(crop_l - ((ExpWidth - CropWidth) / 2))
    crop_r = (int)(crop_r + ((ExpWidth - CropWidth) / 2))
    crop_t = (int)(crop_t - ((ExpHeight - CropHeight) / 2))
    crop_b = (int)(crop_b + ((ExpHeight - CropHeight) / 2))

    # check expanded region
    diff = 0
    if (crop_l < 0):
        diff = 0 - crop_l
        crop_l = crop_l + diff
        crop_r = crop_r + diff
    if (crop_r >= img_width):
        diff = crop_r - (img_width - 1)
        crop_l = crop_l - diff
        crop_r = crop_r - diff

    diff = 0
    if (crop_t < 0):
        diff = 0 - crop_t
        crop_t = crop_t + diff
        crop_b = crop_b + diff
    if (crop_b >= img_height):
        diff = crop_b - (img_height - 1)
        crop_t = crop_t - diff
        crop_b = crop_b - diff

    return (crop_l, crop_t, crop_r, crop_b)


# prepare data class
class Prepare_Data:
    def __init__(self, img_width, img_height, dataset_path) -> None:
        self.image_width = img_width
        self.image_height = img_height
        self.dataset_path = dataset_path

    def prepare_train_data(self, save_url = '', save_img = False):
        # get file count
        img_list = os.listdir(self.dataset_path)
        #print(len(img_list))

        # prepare dataset variables
        train_imgs = []
        train_labels = []

        # prepare data
        CropWidth = self.image_width
        CropHeight = self.image_height
        ExpWidth = EXPAND_WIDTH
        ExpHeight = EXPAND_HEIGHT
        finger_idx = 0
        finger_id = 0
        for item in img_list:
            if item == '.DS_Store':
                continue
            if item == 'README.md':
                continue

            finger_id_tmp = parse_file_name2(item)
            if (int(finger_id_tmp) != finger_id):
                finger_id = int(finger_id_tmp)
                finger_idx = 0

            img_path = self.dataset_path + item

            # get fingerprint region
            (crop_l, crop_t, crop_r, crop_b) = get_fp_region(img_path, CropWidth, CropHeight)

            img = Image.open(img_path)

            # crop for process image
            crop_x = (ExpWidth - CropWidth) / 2
            crop_y = (ExpHeight - CropHeight) / 2
            img = img.crop([crop_l, crop_t, crop_r, crop_b])

            # single crop
            img_c = img.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
            img_arr = np.array(img_c)
            train_imgs.append(img_arr.reshape(CropWidth, CropHeight, 1))
            train_labels.append(finger_id)
            if save_img == True:
                img_path_new = save_url + '{0:05d}'.format(finger_id) + '_' + '{0:02d}'.format(finger_idx) + '.bmp'
                img_c.save(img_path_new)
                finger_idx += 1

            # rotate crop
            for i in range(5):
                ang = random.randint(10, 350)
                img_rot = img.rotate(ang)
                img_c = img_rot.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
                img_arr = np.array(img_c)
                train_imgs.append(img_arr.reshape(CropWidth, CropHeight, 1))
                train_labels.append(finger_id)
                if save_img == True:
                    img_path_new = save_url + '{0:05d}'.format(finger_id) + '_' + '{0:02d}'.format(finger_idx) + '.bmp'
                    img_c.save(img_path_new)
                    finger_idx += 1

            # auto contrast
            for i in range(4):
                ang = random.randint(10, 350)
                img_autocont = ImageOps.autocontrast(img, 20)
                img_rot = img_autocont.rotate(ang)
                img_c = img_rot.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
                img_arr = np.array(img_c)
                train_imgs.append(img_arr.reshape(CropWidth, CropHeight, 1))
                train_labels.append(finger_id)
                if save_img == True:
                    img_path_new = save_url + '{0:05d}'.format(finger_id) + '_' + '{0:02d}'.format(finger_idx) + '.bmp'
                    img_c.save(img_path_new)
                    finger_idx += 1

        train_imgs_arr = np.array(train_imgs)
        train_labels_arr = np.array(train_labels)
        return train_imgs_arr, train_labels_arr

    def prepare_eval_data(self, eval_count = 100, save_url = '', save_img = False):
        # get file count
        img_list = os.listdir(self.dataset_path)
        #print(len(img_list))

        # prepare dataset variables
        eval_imgs = []
        eval_labels = []

        # prepare data
        CropWidth = self.image_width
        CropHeight = self.image_height
        ExpWidth = EXPAND_WIDTH
        ExpHeight = EXPAND_HEIGHT
        finger_id = 0
        for choice_idx in range(eval_count):
            while True:
                item = random.choice(img_list)
                if (item != '.DS_Store') and (item != 'README.md'):
                    break

            finger_id = parse_file_name2(item)
            img_path = self.dataset_path + item

            # get fingerprint region
            (crop_l, crop_t, crop_r, crop_b) = get_fp_region(img_path, CropWidth, CropHeight)

            img = Image.open(img_path)

            # crop for process image
            crop_x = (ExpWidth - CropWidth) / 2
            crop_y = (ExpHeight - CropHeight) / 2
            img = img.crop([crop_l, crop_t, crop_r, crop_b])

            # single crop
            img_c = img.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
            img_arr = np.array(img_c)
            eval_imgs.append(img_arr.reshape(CropWidth, CropHeight, 1))
            eval_labels.append(finger_id)
            if save_img == True:
                img_path_new = save_url + '{0:05d}'.format(choice_idx) + '_' + '{0:02d}'.format(finger_id) + '.bmp'
                img_c.save(img_path_new)

        eval_imgs_arr = np.array(eval_imgs)
        eval_labels_arr = np.array(eval_labels)
        return eval_imgs_arr, eval_labels_arr
