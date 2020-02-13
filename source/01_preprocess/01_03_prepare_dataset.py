# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob, os


def parse_file_name(img_path):
    user_id, _ = os.path.splitext(os.path.basename(img_path))
    #print(user_id)
    return np.array([user_id], dtype=np.uint16)

def parse_file_name2(img_path):
    user_id, _ = os.path.splitext(os.path.basename(img_path))
    user_id, finger_id = user_id.split('_')
    #print(user_id, finger_id)
    return np.array([user_id], dtype=np.uint16)



# %%
# make data
img_list = sorted(glob.glob('../../dataset/train_data/*.bmp'))
print(len(img_list))

imgs = np.empty((len(img_list), 160, 160, 1), dtype = np.uint8)
labels = np.empty((len(img_list), 1), dtype = np.uint16)

for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imgs[i] = img.reshape(160, 160, 1)

    # get user id
    labels[i] = parse_file_name2(img_path)

np.save('../../dataset/np_data/img_train.npy', imgs)
np.save('../../dataset/np_data/label_train.npy', labels)



img_list = sorted(glob.glob('../../dataset/real_data/*.bmp'))
print(len(img_list))

imgs = np.empty((len(img_list), 160, 160, 1), dtype = np.uint8)
labels = np.empty((len(img_list), 1), dtype = np.uint16)

for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imgs[i] = img.reshape(160, 160, 1)

    # get user id
    labels[i] = parse_file_name(img_path)

np.save('./dataset/np_data/img_real.npy', imgs)
np.save('./dataset/np_data/img_real.npy', labels)

print('Complete')


# %%
