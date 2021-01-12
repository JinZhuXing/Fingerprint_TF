# Dataset

This folder contains dataset for training and evaluation.

Here are 4 subfolders as below.

    original
    train_data
    real_data
    np_data


## Original Fingerprint Image (original)

This folder contains original fingerprint images.

The image size must be greater than 160x160.

The image file must be a 8bit grayscale bitmap(.bmp) file.

And the file name must be "<FingerID (5 digits)>_<Index (2 digits)>.bmp"

For example, if dataset contains 3 fingers and each finger has two images, the folder structure will be like below.

    dataset\original
        00000_00.bmp
        00000_01.bmp
        00001_00.bmp
        00001_01.bmp
        00002_00.bmp
        00002_01.bmp


## Fingerprint Image for Training (train_data)

This folder contains fingerprint images for training.

Training images can be generated with "01_01_prepare_train_data.py" file.

This file can generate training images from original images.

    python 01_01_prepare_train_data.py


## Clear Fingerprint Image (real_data)

This folder contains clear fingerprint images.

Clear images can be generated with "01_02_prepare_real_data.py" file.

This file will copy first image for each finger from train_data folder.

    python 01_02_prepare_real_data.py


## Numpy Data (np_data)

This folder contains numpy data for training and evaluation.

Numpy data file can be generated with "01_03_prepare_dataset.py" file.

    python 01_03_prepare_dataset.py


## Sample Dataset

You can get sample dataset from here.

    URL : https://pan.baidu.com/s/1Y4L5xgHNK47BeR9bVbRrqA
    Password : xeqp

This dataset was created from FVC2000_DB4_B.

Please donate fingerprint images for research purpose.

Thanks.
