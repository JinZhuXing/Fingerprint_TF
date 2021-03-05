from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


from prepare import Prepare_Data
from model_net import Model_Net


# global variables
IMAGE_WIDTH_LIMIT = 192
IMAGE_HEIGHT_LIMIT = 192


# data generator definition
class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_data, label_data, img_width, img_height, batch_size = 32, shuffle = True):
        'Initialization'
        self.img_data = img_data
        self.label_data = label_data
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_data) / self.batch_size) * 2)

    def __getitem__(self, index):
        'Generate one batch of data'
        real_idx = index
        index = int(np.floor(index / 2))
        img1_batch = self.img_data[index * self.batch_size : (index + 1) * self.batch_size]
        label1_batch = self.label_data[index * self.batch_size : (index + 1) * self.batch_size]
        img2_batch = np.empty((self.batch_size, self.img_width, self.img_height, 1), dtype = np.float32)
        label2_batch = np.zeros((self.batch_size, 1), dtype = np.float32)

        for i, idx in enumerate(label1_batch):
            if random.random() > 0.5:
                # put matched image
                while True:
                    match_idx = random.choice(range(len(self.label_data)))
                    if (self.label_data[match_idx] == idx):
                        break

                img2_batch[i] = self.img_data[match_idx]
                label2_batch[i] = 1.
            else:
                # put unmatched image
                while True:
                    unmatch_idx = random.choice(range(len(self.label_data)))
                    if (self.label_data[unmatch_idx] != idx):
                        break

                img2_batch[i] = self.img_data[unmatch_idx]
                label2_batch[i] = 0.
                
        index = real_idx
        if (index < int(np.floor(len(self.img_data) / self.batch_size))):
            return [img1_batch.astype(np.float32) / 255., img2_batch.astype(np.float32) / 255.], label2_batch
        
        return [img2_batch.astype(np.float32) / 255., img1_batch.astype(np.float32) / 255.], label2_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            self.img_data, self.label_data = shuffle(self.img_data, self.label_data)


# main process
def main(args):
    image_width = args.image_width
    image_height = args.image_height
    dataset_path = args.dataset_path
    check_point = args.check_point
    save_model = args.save_model
    save_model_path = args.save_model_path
    train_epoch = args.train_epoch

    # check parameter
    if ((image_width > IMAGE_WIDTH_LIMIT) or (image_height > IMAGE_HEIGHT_LIMIT)):
        print('Image size must be smaller than', IMAGE_WIDTH_LIMIT, 'x', IMAGE_HEIGHT_LIMIT)
        return
    
    # show tensorflow information
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU: ", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
    
    # show information
    print('Train start')
    print('\tImage Width: ', image_width)
    print('\tImage Height: ', image_height)
    print('\tDataset Path: ', dataset_path)
    print('\tCheckpoint Path: ', check_point)
    print('\tTrain Epoch: ', train_epoch)

    # create a callback that saves the model's weights
    checkpoint_path = check_point + 'cp_' + str(image_width) + '_' + str(image_height) + '.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1)

    # prepare model
    model_net = Model_Net(image_width, image_height)
    model = model_net.build_model()
    if (os.path.exists(checkpoint_path + '.index')):
        print('continue training')
        model.load_weights(checkpoint_path)
    
    # check only save
    if (save_model == 1):
        model_path = save_model_path + 'fp' + str(image_width) + '_' + str(image_height) + '.h5'
        model.save(model_path)
        return

    # prepare data
    while True:
        print('Prepare Dataset...')
        train_data_pre = Prepare_Data(image_width, image_height, dataset_path)
        img_data, label_data = train_data_pre.prepare_train_data()
        print('Finished: ', img_data.shape, label_data.shape)

        # split data
        print('Split Dataset for train and validation...')
        img_train, img_val, label_train, label_val = train_test_split(img_data, label_data, test_size = 0.1)
        print('Finished: ')
        print(img_train.shape, label_train.shape)
        print(img_val.shape, label_val.shape)

        # prepare data generator
        train_gen = DataGenerator(img_train, label_train, image_width, image_height, shuffle = True)
        val_gen = DataGenerator(img_val, label_val, image_width, image_height, shuffle = True)

        # training model
        model.fit(train_gen, epochs = train_epoch, validation_data = val_gen, callbacks = [cp_callback])

        # save model
        model_path = save_model_path + 'fp' + str(image_width) + '_' + str(image_height) + '.h5'
        model.save(model_path)


# argument parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # image size information
    parser.add_argument('--image_width', type = int,
        help = 'Process image width', default = 128)
    parser.add_argument('--image_height', type = int,
        help = 'Process image height', default = 128)
    parser.add_argument('--dataset_path', type = str,
        help = 'Path to fingerprint image dataset', default = '../../dataset/original/')
    parser.add_argument('--check_point', type = str,
        help = 'Path to model checkpoint', default = '../../model/checkpoint/')
    parser.add_argument('--save_model', type = int,
        help = 'Only save model from checkpoint', default = 0)
    parser.add_argument('--save_model_path', type = str,
        help = 'Path to model', default = '../../model/result/')
    parser.add_argument('--train_epoch', type = int,
        help = 'Train epoch count', default = 100)

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))