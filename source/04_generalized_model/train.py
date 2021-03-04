from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


# main process
def main(args):
    image_width = args.image_width
    image_height = args.image_height
    dataset_path = args.dataset_path
    
    # show information
    print('Train start')
    print('\tImage Width: ', image_width)
    print('\tImage Height: ', image_height)
    print('\tDataset Path: ', dataset_path)


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

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))