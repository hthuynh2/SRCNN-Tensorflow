"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2

import tensorflow as tf

try:
    xrange
except:
    xrange = range

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def preprocess(image_path, label_path="", scale=2):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    label = None
    image = imread(image_path)
    image = image / 255.

    if label_path != "":
        label = imread(label_path)
        label = label / 255.

    input_ = scipy.ndimage.interpolation.zoom(image, (scale / 1.), prefilter=False)

    return input_, label


def get_image_path(is_train, s, num):
    assert (s == 128 or s == 64)
    path = os.path.join(os.getcwd(), "xray_images/")
    image_name = ""
    if not is_train:
        path += 'test_images_'
        image_name += 'test_'
    else:
        path += 'train_images_'
        image_name += 'train_'
    if s == 64:
        path += '64x64'
    elif s == 128:
        path += '128x128'
    num_str = format(num, "05")
    image_name += num_str + ".png"
    return path + "/" + image_name


# def prepare_data():
#     """
#     Args:
#       dataset: choose train dataset or test dataset
#
#       For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
#     """
#     inputs_paths = None
#     labels_paths = None
#     if FLAGS.is_train:
#         inputs_dir = os.path.join(os.getcwd(), "xray_images/test_images_64x64/")
#         labels_dir = os.path.join(os.getcwd(), "xray_images/test_images_128x128/")
#         for i in range(4000, 20000):
#             input_path = inputs_dir + "train_" + format(i, "05") + ".png"
#             label_path = labels_dir + "train_" + format(i, "05") + ".png"
#
#         filenames = os.listdir(dataset)
#         data_dir = os.path.join(os.getcwd(), dataset)
#         data_path = glob.glob(os.path.join(data_dir, "*.png"))
#     else:
#         data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
#         data_path = glob.glob(os.path.join(data_dir, "*.png"))
#
#     return data_path


def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)


# def modcrop(image, scale=3):
#     """
#     To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
#
#     We need to find modulo of height (and width) and scale factor.
#     Then, subtract the modulo from height (and width) of original image size.
#     There would be no remainder even after scaling operation.
#     """
#     if len(image.shape) == 3:
#         h, w, _ = image.shape
#         h = h - np.mod(h, scale)
#         w = w - np.mod(w, scale)
#         image = image[0:h, 0:w, :]
#     else:
#         h, w = image.shape
#         h = h - np.mod(h, scale)
#         w = w - np.mod(w, scale)
#         image = image[0:h, 0:w]
#     return image


def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    # if config.is_train:
    #   data = prepare_data(sess, True)
    # else:
    #   data = prepare_data(sess, dataset="Test")
    # data = prepare_data(sess, config.is_train)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in xrange(4000, 20000):
            input_path = get_image_path(config.is_train, 64, i)
            label_path = get_image_path(config.is_train, 128, i)
            input_, label_ = preprocess(input_path, label_path, config.scale)

            h, w = input_.shape

            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                    sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                                y + int(padding):y + int(padding) + config.label_size]  # [21 x 21]

                    # Make channel value
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:
        for i in range(1, 2):
            input_path = get_image_path(config.is_train, 64, i)
            input_, _ = preprocess(input_path, scale=config.scale)

            h, w = input_.shape

            # Numbers of sub-images in height and width of image are needed to compute merge operation.
            nx = ny = 0
            for x in range(0, h - config.image_size + 1, config.stride):
                nx += 1
                ny = 0
                for y in range(0, w - config.image_size + 1, config.stride):
                    ny += 1
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                    # sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                    #             y + int(padding):y + int(padding) + config.label_size]  # [21 x 21]

                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    # sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    # sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    if config.is_train:
        # make_data(sess, arrdata, arrlabel)
        print("Set up input succesfully!")
        return arrdata, arrlabel
    if not config.is_train:
        arrlabel = np.zeros((1, 16, 16, 1))
        # make_data(sess, arrdata, arrlabel)
        return nx, ny, arrdata, arrlabel


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
