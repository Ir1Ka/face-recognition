#!/usr/bin/env python3
# coding: utf-8

'''
Image data preprocessing using TensorFlow
'''

__author__ = 'IriKa'

import tensorflow as tf
# *NOTE*: Some OP does not run on GPU and only intead of CPU.
# Therefore, we need config `allow_soft_placement=True` to `tf.Session()`.
# It will be able to make the OP running on the GPU not supported and automatically run on the CPU.

class preprocessing_for_image:
    '''We assume that the entire image is a human face.

    *Please use other algorithms to detect and crop faces.*
    '''

    def __init__(self, in_data, train, out_size=None, normalization=True):
        '''Init.

        Args:
            in_data: A 4-dimension tensor.
            train: A `bool` type tensor. It means that current is training or testing.
            out_size: A 2 `int` elements `tuple` or `list`. Default same as `in_data` size.
            normalization: Normalization for outputs. Default is `True`.

        NOTE: If the `in_size` and `out_size` are not in the same proportion,
            it will cause the image after processing to be distorted.
        '''
        self.scope_name = 'preprocessing'
        self.in_data = in_data
        self.train_ph = train
        self.in_size = in_data.shape[1:3]
        if out_size is None:
            out_size = self.in_size
        self.out_size = out_size
        self.normalization = normalization
        self.__gen_placeholder()
        self.__hyperparameter()
        self.__preprocess()
 
    def __gen_placeholder(self):
        '''Generate some Place Holder for preprocessing.
        '''
        self.rotate_angles_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='rotate_angles')
        #self.color_ordering_ph = tf.placeholder(dtype=tf.uint8, shape=(0,), name='color_ordering')

    def __hyperparameter(self):
        '''Define some hyperparameter.
        '''
        self.JPEG_QUALITY = (80, 100)
        # The value of `BRIGHTNESS` from cifar10.
        self.BRIGHTNESS = 63
        self.SATURATION = (0.5, 1.5)
        self.HUE = 0.2
        self.CONTRAST = (0.5, 1.5)

    def image_transformation(self, imgs):
        '''Random transformation for the input images.

        * Random flip left/right.
        * Random rotate, use a placeholder for random.
        * Random image quality.
        '''
        with tf.variable_scope('image_transformation') as scope:
            # Whether to do the flip, you need to do more research.
            imgs = tf.image.random_flip_left_right(imgs)
            # The angle of rotation is in radians.
            imgs = tf.contrib.image.rotate(imgs, self.rotate_angles_ph)
            # Temporarily not randomly adjusting the image quality,
            # it may be due to usage or tf bugs, and the image quality cannot be adjusted in batches.
            #imgs = tf.image.random_jpeg_quality(imgs, self.JPEG_QUALITY[0], self.JPEG_QUALITY[1])
        return imgs

    def distort_color(self, imgs):
        '''Random distort the color for the input images.

        * Random brightness.
        * Random saturation.
        * Random hue.
        * Random contrast.
        '''
        with tf.variable_scope('distort_color') as scope:
            imgs = tf.image.random_brightness(imgs, max_delta=self.BRIGHTNESS)
            imgs = tf.image.random_saturation(imgs, lower=self.SATURATION[0], upper=self.SATURATION[1])
            imgs = tf.image.random_hue(imgs, max_delta=self.HUE)
            imgs = tf.image.random_contrast(imgs, lower=self.CONTRAST[0], upper=self.CONTRAST[1])
        return imgs

    def data_standardization(self, imgs):
        '''Standardize the input image batch data.

        ** NOTE: the `imgs.dtype` must be about `tf.float`.
        * Remove the DC component.
        * Adjust the variance to 1.
        '''
        with tf.variable_scope('normalization') as scope:
            axis = list(range(len(imgs.get_shape()) - 1))
            mean, variance = tf.nn.moments(imgs, axis)
            bn = tf.nn.batch_normalization(imgs, mean, variance, 0., 1., 1e-3)
        return bn

    def get_placeholder(self):
        '''Get the generated Place Holder.
        '''
        return self.rotate_angles_ph

    def get_output(self):
        '''Get the output tensor op.
        '''
        return self.output

    def __preprocess(self):
        '''Generate the preprocess using tensorflow.

        Using above preprocessing method.
        '''
        with tf.variable_scope(self.scope_name) as scope:
            # Adjusting the resolution first will help reduce the amount of calculations.
            resized_imgs = tf.image.resize_images(self.in_data, size=self.out_size)
            imgs = tf.cast(resized_imgs, dtype=tf.float32)
            train_fn = lambda: self.distort_color(self.image_transformation(imgs))
            test_fn = lambda : imgs
            imgs = tf.cond(self.train_ph, train_fn, test_fn)
            if self.normalization is True:
                imgs = self.data_standardization(imgs)
        self.output = imgs

def main(face_data):
    PI = 3.1415
    batch_size = 1000
    # The angle of rotation is in radians.
    rotate_angles_max_delta = 15 / 180 * PI

    in_ph = tf.placeholder(dtype=tf.uint8, shape=(None, 250, 250, 3), name='in_data')
    train_ph = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')
    imgs_preprocess = preprocessing_for_image(in_ph, train_ph)
    rotate_angles_ph = imgs_preprocess.get_placeholder()
    with tf.device('/device:GPU:0'):
        out = imgs_preprocess.get_output()
    # This configuration is required, otherwise an error will be run, for the reason of the beginning of the file.
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.as_default()
        for i in range(5):
            print('Index:', i)
            batch, lable = face_data.next_batch(batch_size=batch_size)
            print('name:', lable[0])
            rotate_angles = np.random.uniform(low=-rotate_angles_max_delta, high=rotate_angles_max_delta, size=batch_size)
            print('rotate angle:', rotate_angles[0])
            feed_dict = {in_ph: batch, train_ph: True, rotate_angles_ph: rotate_angles}
            preprocessed = out.eval(feed_dict=feed_dict)
            '''
            # To show the preprocessed images.
            max = np.max(preprocessed[0, ...], axis=(1,2), keepdims=True)
            min = np.min(preprocessed[0, ...], axis=(1,2), keepdims=True)
            plt.imshow((preprocessed[0, ...] - min) / (max - min))
            plt.show()
            '''
            print('type:', type(preprocessed), 'dtype:', preprocessed.dtype, 'shape', preprocessed.shape, end='\n\n')

if __name__ == '__main__':
    # Test it
    import numpy as np
    import matplotlib.pylab as plt
    from casia_webface import casia_webface

    with casia_webface() as face_data:
        main(face_data)
