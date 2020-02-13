# %%
import os
import shutil


# set directory
original_dataset_dir = '../../dataset/train_data/'
result_dataset_dir = '../../dataset/real_data/'

# copy real data
for i in range(10000):
    img_path = original_dataset_dir + '{0:05d}'.format(i) + '_00.bmp'
    img_path_new = result_dataset_dir + '{0:05d}'.format(i) + '.bmp'
    if (os.path.exists(img_path) == False):
        break
    shutil.copy(img_path, img_path_new)



# %%
