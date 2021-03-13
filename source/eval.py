from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np

import tensorflow as tf


from prepare import Prepare_Data
from model_net import Model_Net


# global variables
IMAGE_WIDTH_LIMIT = 192
IMAGE_HEIGHT_LIMIT = 192


# main process
def main(args):
    image_width = args.image_width
    image_height = args.image_height
    dataset_path = args.dataset_path
    check_point = args.check_point
    eval_count = args.eval_count

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
    print('\tEval Count: ', eval_count)

    # prepare model
    model_net = Model_Net(image_width, image_height)
    model = model_net.build_model()
    checkpoint_path = check_point + 'cp_' + str(image_width) + '_' + str(image_height) + '.ckpt'
    if (os.path.exists(checkpoint_path + '.index')):
        model.load_weights(checkpoint_path)
    else:
        print('Can not load checkpoint')
        return

    # prepare data
    print('Prepare Dataset...')
    eval_data_pre = Prepare_Data(image_width, image_height, dataset_path)
    img_data, label_data = eval_data_pre.prepare_eval_data(eval_count = eval_count)
    print('Finished: ', img_data.shape, label_data.shape)

    # evaluation
    total_count = 0
    error_reject_cnt = 0
    error_accept_cnt = 0
    error_rage = 0.8
    for input_idx in range(label_data.shape[0]):
        print('Processing #', input_idx, '')
        for db_idx in range(label_data.shape[0]):
            input_img = img_data[input_idx].reshape((1, image_width, image_height, 1)).astype(np.float32) / 255.
            db_img = img_data[db_idx].reshape((1, image_width, image_height, 1)).astype(np.float32) / 255.
            pred_right = model.predict([input_img, db_img])
            if (label_data[input_idx] == label_data[db_idx]):
                if (pred_right < error_rage):
                    print('False Reject = ', pred_right)
                    error_reject_cnt += 1
            else:
                if (pred_right > error_rage):
                    print('False Accept = ', pred_right, ', ID = ', db_idx)
                    error_accept_cnt += 1
            total_count += 1

    # show result
    print('Evaluation Finished')
    print('Total Count = ', total_count)
    print('Error Reject Count = ', error_reject_cnt)
    print('Error Accept Count = ', error_accept_cnt)
    print('Error Rate = ', ((error_reject_cnt + error_accept_cnt) / total_count) * 100)


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
    parser.add_argument('--check_point', type = str,
        help = 'Path to model checkpoint', default = '../model/checkpoint/')
    parser.add_argument('--eval_count', type = int,
        help = 'Evaluation data count', default = 100)

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))