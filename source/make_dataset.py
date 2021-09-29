from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

from prepare import Prepare_Data


# global variables
IMAGE_WIDTH_LIMIT = 192
IMAGE_HEIGHT_LIMIT = 192


# main process
def main(args):
    image_width = args.image_width
    image_height = args.image_height
    dataset_path = args.dataset_path
    out_path = args.out_path

    # check parameter
    if ((image_width > IMAGE_WIDTH_LIMIT) or (image_height > IMAGE_HEIGHT_LIMIT)):
        print('Image size must be smaller than', IMAGE_WIDTH_LIMIT, 'x', IMAGE_HEIGHT_LIMIT)
        return
    
    # show information
    print('Train start')
    print('\tImage Width: ', image_width)
    print('\tImage Height: ', image_height)
    print('\tDataset Path: ', dataset_path)
    print('\tOutput Path: ', out_path)

    # prepare data
    print('Prepare Dataset...')
    train_data_pre = Prepare_Data(image_width, image_height, dataset_path)
    img_data, label_data = train_data_pre.prepare_train_data()
    print('Finished: ', img_data.shape, label_data.shape)

    # split data
    print('Split Dataset for train and validation...')
    img_proc, img_test, label_proc, label_test = train_test_split(img_data, label_data, test_size = 0.1)
    img_train, img_val, label_train, label_val = train_test_split(img_proc, label_proc, test_size = 0.1)
    print('Finished: ')
    print(img_train.shape, label_train.shape)
    print(img_val.shape, label_val.shape)
    print(img_test.shape, label_test.shape)

    # save data as numpy
    np.save(out_path + 'img_train.npy', img_train)
    np.save(out_path + 'label_train.npy', label_train)
    np.save(out_path + 'img_val.npy', img_val)
    np.save(out_path + 'label_val.npy', label_val)
    np.save(out_path + 'img_test.npy', img_test)
    np.save(out_path + 'label_test.npy', label_test)


# argument parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # image size information
    parser.add_argument('--image_width', type = int,
        help = 'Process image width', default = 128)
    parser.add_argument('--image_height', type = int,
        help = 'Process image height', default = 128)
    parser.add_argument('--dataset_path', type = str,
        help = 'Path to fingerprint image dataset', default = '../dataset/original/')
    parser.add_argument('--out_path', type = str,
        help = 'Path to output dataset', default = '../dataset/')

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))