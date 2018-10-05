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
    '''
    We assume that the entire image is a human face.
    *Please use other algorithms to detect and crop faces.*
    '''

    def __init__(self, in_data, out_size=None):
        '''
        Init.
        NOTE: If the `in_size` and `out_size` are not in the same proportion,
            it will cause the image after processing to be distorted.

        @Args
            in_size: a 2 elements tuple.
        '''
        self.in_data = in_data
        self.in_size = in_data.shape[1:3]
        if out_size is None:
            out_size = self.in_size
        self.out_size = out_size
        #print('in_size', self.in_size)
        #print('out_size', self.out_size)
        self.__gen_placeholder()
        self.__hyperparameter()
 
    def __gen_placeholder(self):
        '''
        Generate some Place Holder for preprocessing.
        '''
        self.is_train_ph = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
        self.rotate_angles_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='rotate_angles')
        #self.color_ordering_ph = tf.placeholder(dtype=tf.uint8, shape=(0,), name='color_ordering')

    def __hyperparameter(self):
        '''
        Define some hyperparameter.
        '''
        self.JPEG_QUALITY = (80, 100)
        self.BRIGHTNESS = 16./255.
        self.SATURATION = (0.5, 1.5)
        self.HUE = 0.2
        self.CONTRAST = (0.5, 1.5)

    def image_transformation(self, imgs):
        '''
        * Radom flip left/right.
        * Random rotate, use a placeholder for random.
        * Random image quality.
        '''
        # Whether to do the flip, you need to do more research.
        imgs = tf.image.random_flip_left_right(imgs)
        # The angle of rotation is in radians.
        imgs = tf.contrib.image.rotate(imgs, self.rotate_angles_ph)
        # Temporarily not randomly adjusting the image quality,
        # it may be due to usage or tf bugs, and the image quality cannot be adjusted in batches.
        #imgs = tf.image.random_jpeg_quality(imgs, self.JPEG_QUALITY[0], self.JPEG_QUALITY[1])
        return imgs

    def distort_color(self, imgs):
        '''
        * Random brightness.
        * Random saturation.
        * Random hue.
        * Random contrast.
        '''
        imgs = tf.image.random_brightness(imgs, max_delta=self.BRIGHTNESS)
        imgs = tf.image.random_saturation(imgs, lower=self.SATURATION[0], upper=self.SATURATION[1])
        imgs = tf.image.random_hue(imgs, max_delta=self.HUE)
        imgs = tf.image.random_contrast(imgs, lower=self.CONTRAST[0], upper=self.CONTRAST[1])
        return imgs

    def data_standardization(self, imgs):
        '''
        ** NOTE: the `imgs.dtype` must be about `tf.float`.
        * Remove the DC component.
        * Adjust the variance to 1.
        '''
        mean = tf.reduce_mean(imgs, axis=(1,2), keepdims=True)
        std = tf.keras.backend.std(imgs, axis=(1,2), keepdims=True)
        return (imgs - mean) / std

    def get_placeholder(self):
        '''
        Get the generated Place Holder.
        '''
        return self.is_train_ph, self.rotate_angles_ph

    def get_output(self):
        '''
        Get the output tensor op using above preprocessing method.
        '''
        # Adjusting the resolution first will help reduce the amount of calculations.
        resized_imgs = tf.image.resize_images(self.in_data, size=self.out_size)
        imgs = tf.cast(resized_imgs, dtype=tf.float32)
        train_fn = lambda: self.distort_color(self.image_transformation(imgs))
        test_fn = lambda : imgs
        imgs = tf.case([(tf.equal(self.is_train_ph, True), train_fn)], default=test_fn)
        imgs = self.data_standardization(imgs)
        return imgs

def main(face_data):
    PI = 3.1415
    batch_size = 1000
    # The angle of rotation is in radians.
    rotate_angles_max_delta = 15 / 180 * PI

    in_ph = tf.placeholder(dtype=tf.uint8, shape=(None, 250, 250, 3), name='in_data')
    imgs_preprocess = preprocessing_for_image(in_ph)
    is_train_ph, rotate_angles_ph = imgs_preprocess.get_placeholder()
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
            #plt.imshow(batch[0, ...])
            #plt.show()
            rotate_angles = np.random.uniform(low=-rotate_angles_max_delta, high=rotate_angles_max_delta, size=batch_size)
            print('rotate angle:', rotate_angles[0])
            feed_dict = {in_ph: batch, is_train_ph: True, rotate_angles_ph: rotate_angles}
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
