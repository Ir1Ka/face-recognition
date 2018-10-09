#!/usr/bin/env python3
# coding: utf-8

'''
Auto Encoder Neural Networks
'''

__author__ = 'IriKa'

import tensorflow as tf
import tools

class stack_autoencoder:
    '''A NN AutoEncoder.
    '''
    def __init__(self, in_data, layer_num, hidden_outputs):
        '''Constructor

        Args:
            in_data: A 4-D `Tensor`. Note, the last 3-D shape must be known and cannot be `None`.
            layer_num: A `int`. Indicates the number of layers of the `stack_autoencoder`.
            hidden_outputs: A list of hidden layers output feature maps.

        Raises:
            ValueError: If the length of the `hidden_outputs` is not equal `layer_num`.
        '''
        shape = in_data.shape
        if None in shape[-3:]:
            raise ValueError('The last 3-D shape must be known and cannot be None.')
        if len(hidden_outputs) != layer_num:
            raise ValueError('The length of the hidden_outputs must be equal layer_num.')
        self.in_data = in_data
        self.in_shape = self.in_data.shape
        self.layer_num = layer_num
        # The first element is the channels of input layer.
        self.hidden_outputs = self.in_shape[-1:] + hidden_outputs

    def codec(imgs, filter, is_encode,
              stddev=5e-2,
              name=None,
              new_size=None,
              ksize=[1, 2, 2, 1],
              strides=[1, 2, 2, 1]):
        '''Single layer encoder and decoder.

        Args:
            imgs: Images for encode or decode.
            filter: The convolution kernel shape.
            is_encode: `True` indicates that it is an encoder, and `False` indicates that it is a decoder.
            stddev: The convolution kernel initializes the standard deviation.
            name: Name of the codec. Default `encoder` if `is_encode` is `True`, else is `decoder`.
            new_size: The size of the decoder output. If `is_encode` is `False`, the value must be set.
            ksize: Downsampling ksize.
            strides: Downsampling strides.

        Returns:
            Codec output.

        Raises:
            ValueError: If `is_encode` is not `bool`.
                And if `new_size` is `None` when `is_encode` is `False`.
        '''
        if not isinstance(is_encode, bool):
            raise ValueError('The value of the is_encode must be a bool type.')

        if name is None:
            if is_encode:
                name = 'encoder'
            else:
                name = 'decoder'

        layers = [imgs]

        with tf.variable_scope(name) as scope:
            if not is_encode:
                # Upsampling.
                if new_size is None:
                    raise ValueError('The value of the new_size must be set if the is_encode is False.')
                unsampling = tf.image.resize_nearest_nerghbor(layers[-1], new_size, name='upsample')
                layers.append(unsampling)

            # Convolution.
            kernel = tools.variable_with_weight_decay('weights',
                                                      shape=filter,
                                                      stddev=stddev,
                                                      wd=None)
            conv = tf.nn.conv2d(layers[-1], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tools.variable_on_cpu('biased', filter[-1:], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            # Activation.
            activation = tf.nn.relu(pre_activation, name=scope.name)
            layers.append(activation)

            tools.activation_summary(layers[-1])

            if is_encode:
                # Downsampling.
                downsampling = tf.nn.max_pool(layers[-1], ksize=ksize,
                                         strides=strides, padding='SAME', name='downsample')
                layers.append(downsampling)
        return layers[-1]

    def gen_model(self, filter_sizes=[3, 3]):
        if len(filter_sizes) != 1 or len(filter_sizes) != self.layer_num:
            raise ValueError('The length of filter_sizes must be equal 1 or layer_num.')
        self.layer_train_ph = tf.placeholder(name='layer_train', shape=(), dtype=tf.uint8)
        zero_constant = tf.constant(0.0, dtype=tf.float32, shape=(), name='zero_constant')
        if len(filter_sizes) == 1:
            filter_sizes = filter_sizes * self.layer_num
        size_each_hidden = [self.in_shape[-3:-1]]
        layers = [self.in_data]

        # Encode
        with tf.variable_scope('encode') as scope:
            for i in range(self.layer_num):
                name = 'encode_%d' % i+1
                filter = filter_sizes[i] + layers[-1].shape[-1:] + [self.hidden_outputs[i+1]]
                layer = layers[-1]
                if i != 0:
                    include_fn = lambda: layers[-1]
                    exclude_fn = lambda: zero_constant
                    layer = tf.cond(tf.less(i, self.layer_train_ph), include_fn, exclude_fn)
                hidden = codec(layer, filter, True, name=name)
                size_each_hidden.append(hidden.shape[-3:-1])
                layers.append(hidden)

        # Here is encode output.
        self.encoded = layers[-1]

        # Decode
        with tf.variable_scope('decode') as scope:
            for i in range(self.layer_num-1, -1, -1):
                name = 'decode_%d' % i+1
                filter = filter_sizes[i] + layers[-1].shape[-1:] + [self.hidden_outputs[i]]
                layer = layers[-1]
                if i != 0:
                    include_fn = lambda: layers[-1]
                    exclude_fn = lambda: layers[i]
                    layer = tf.cond(tf.less(i, self.layer_train_ph), include_fn, exclude_fn)
                hidden = codec(layer, filter, False, name=name, new_size=size_each_hidden[i])
                layers.append(hidden)

        # the net output - decoded.
        self.decoded = layers[-1]
        return self.decoded

    def get_ph(self):
        return self.layer_train_ph

    def get_encoded(self):
        return self.encoded

    def get_decoded(self):
        return self.decoded

def main():
    pass

if __name__ == "__main__":
    import numpy as np
    from casia_webface import casia_webface
    from preprocessing import preprocessing_for_image
    # For display
    from matplotlib import pyplot as plt
    main()

