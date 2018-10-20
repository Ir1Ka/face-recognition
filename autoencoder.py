#!/usr/bin/env python3
# coding: utf-8

'''
Auto Encoder Neural Networks
'''

__author__ = 'IriKa'

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

import tools

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
            name = 'encode'
        else:
            name = 'decode'

    layers = [imgs]

    with tf.variable_scope(name) as scope:
        if not is_encode:
            # Upsampling.
            if new_size is None:
                raise ValueError('The value of the new_size must be set if the is_encode is False.')
            unsampling = tf.image.resize_nearest_neighbor(layers[-1], new_size, name='upsample')
            layers.append(unsampling)

        # Convolution.
        kernel = tools.variable_with_weight_decay('weights',
                                                  shape=filter,
                                                  stddev=stddev,
                                                  wd=None)
        conv = tf.nn.conv2d(layers[-1], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tools.variable_on_cpu('biases', filter[-1:], tf.zeros_initializer)
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

def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.shape.as_list()[-1:]
        moving_mean = tools.variable_on_cpu('mean', params_shape, tf.zeros_initializer, trainable=False)
        moving_variance = tools.variable_on_cpu('variance', params_shape, tf.ones_initializer, trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tools.variable_on_cpu('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tools.variable_on_cpu('gamma', params_shape, initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

class stack_autoencoder:
    '''A NN AutoEncoder.
    '''
    def __init__(self, in_data, layer_num, hidden_outputs, train):
        '''Constructor

        Args:
            in_data: A 4-D `Tensor`. Note, the last 3-D shape must be known and cannot be `None`.
            layer_num: A `int`. Indicates the number of layers of the `stack_autoencoder`.
            hidden_outputs: A list of hidden layers output feature maps.
            train: A `bool` type tensor. It means that current is training or testing.

        Raises:
            ValueError: If the length of the `hidden_outputs` is not equal `layer_num`.
        '''
        shape = in_data.shape
        if None in shape[-3:]:
            raise ValueError('The last 3-D shape must be known and cannot be None.')
        if len(hidden_outputs) != layer_num:
            raise ValueError('The length of the hidden_outputs must be equal layer_num.')

        self.scope_name = 'stack_autoencoder'
        self.in_data = in_data
        self.in_shape = tf.shape(self.in_data)
        self.layer_num = layer_num
        # The first element is the channels of input layer.
        in_channal = self.in_data.shape.as_list()[-1]
        if in_channal is None:
            raise ValueError('The last dimension of input data must be known.')
        self.hidden_outputs = [in_channal] + hidden_outputs
        self.train_ph = train

    def model(self, filter_sizes=[[3, 3]],
                  ksize=[[1, 2, 2, 1]],
                  strides=[[1, 2, 2, 1]]):
        '''A wrapper that generates the model function.

        Args:
            filter_sizes:   The sizes of the convolution kernel. The default if `[3, 3]`, but you can also customiz it.
                You can only provide one size so that all convolution kernel use the same size.
                Well, you can provide size for each layer of convolution kernel.
                *NOTE*: The convolution kernel of the encoder and decoder of the corresponding layer will use the same size.
            ksize: The encode down sample ksize (max_pool).
            strides: The encode down sample strides (max_pool).

        Returns:
            The final decoder output.
            Well, the output of the hidden layer will also be stored.
        '''
        with tf.variable_scope(self.scope_name) as scope:
            return self.__model(filter_sizes, ksize, strides)

    def __model(self, filter_sizes, ksize, strides):
        '''A generates the model function.

        Args:
            filter_sizes: The sizes of the convolution kernel.
                See the `stack_autoencoder.model` for more information.
            ksize: The encode down sample ksize (max_pool).
            strides: The encode down sample strides (max_pool).

        Returns:
            The final decoder output.
            Well, the output of the hidden layer will also be stored.
        '''
        if len(filter_sizes) != 1 and len(filter_sizes) != self.layer_num:
            raise ValueError('The length of filter_sizes must be equal 1 or layer_num.')
        if len(ksize) != 1 and len(ksize) != self.layer_num:
            raise ValueError('The length of ksize must be equal 1 or layer_num.')
        if len(strides) != 1 and len(strides) != self.layer_num:
            raise ValueError('The length of strides must be equal 1 or layer_num.')

        if len(filter_sizes) == 1:
            filter_sizes = filter_sizes * self.layer_num
        if len(ksize) == 1:
            ksize = ksize * self.layer_num
        if len(strides) == 1:
            strides = strides * self.layer_num

        self.layer_train_ph = tf.placeholder(name='layer_train', shape=(), dtype=tf.int32)
        zero_constant = tf.constant(0.0, dtype=tf.float32, shape=(), name='zero_constant')
        size_each_hidden = [self.in_shape[-3:-1]]
        layers = [self.in_data]

        # All of the encode outputs will store here.
        self.encoded_list = [self.in_data]

        # Encode
        with tf.variable_scope('encoder') as scope:
            for i in range(self.layer_num):
                filter = filter_sizes[i] + self.hidden_outputs[i: i+2]
                layer = layers[-1]
                with tf.variable_scope('hidden_%d' % (i+1)) as scope:
                    if i != 0:
                        include_fn = lambda var=layers[-1]: var
                        exclude_fn = lambda var=layers[-1]: tf.fill(tf.shape(var), 0.0, name='zero')
                        layer = tf.cond(tf.less(i, self.layer_train_ph), include_fn, exclude_fn)
                        layer = batch_norm(layer, self.train_ph, name='BN')
                    encode = codec(layer, filter, True)
                    size_each_hidden.append(tf.shape(encode)[-3:-1])
                layers.append(encode)
                self.encoded_list.append(encode)

        # Decode
        with tf.variable_scope('decoder') as scope:
            for i in range(self.layer_num-1, -1, -1):
                filter = filter_sizes[i] + self.hidden_outputs[i-self.layer_num: i-self.layer_num-2: -1]
                with tf.variable_scope('hidden_%d' % (i+1)) as scope:
                    bn = batch_norm(layers[-1], self.train_ph, name='BN')
                    decode = codec(bn, filter, False, new_size=size_each_hidden[i])
                    hidden = decode
                    if i != 0:
                        include_fn = lambda var=decode: var
                        exclude_fn = lambda var=layers[i]: var
                        hidden = tf.cond(tf.less(i, self.layer_train_ph), include_fn, exclude_fn)
                layers.append(hidden)

        # the net output - decoded.
        self.decoded = layers[-1]
        return self.decoded

    def get_ph(self):
        ''' Get placeholder of the Stack AutoEncoder.
        '''
        return self.layer_train_ph

    def get_encoded(self, index=None):
        ''' Get encoded output of the AutoEncoder.

        The hidden layer coded output can be used to make sparse penalties during network training.
        Or visualize.

        Args:
            index: A `int`. The default is equal to `layer_num`. `0` means to get input of the AutoEncoder.
                The value must be less than `layer_num`.

        Raises:
            ValueError: If `index` is greater than `layer_num`.
        '''
        if index > self.layer_num:
            raise ValueError('The index must be less than the layer_num.')

        if index is None:
            index = self.layer_num
        return self.encoded_list[index]

    def get_decoded(self):
        ''' Get decoded output of the Stack AutoEncoder.
        '''
        return self.decoded

    def get_variable_for_layer(self, index, trainable=None):
        ''' Get variable of the Stack AutoEncoder.

        Get the variables of the specified layer, which is convenient for Stack AutoEncoder training.

        Args:
            index: A `int`. Specifies which layer of variables to retrieve,
                including the corresponding encoder layer and decoder layer.
            trainable: A `bool`. Indicates whether it is a trainingable variable. The `True` and default are trainable.
        '''
        var_list = []
        hidden_name = 'encoder/hidden_%d' % index, 'decoder/hidden_%d' % index
        if trainable is True or trainable is None:
            all_vars = tf.trainable_variables(scope=self.scope_name)
        else:
            all_vars = tf.global_variables(scope=self.scope_name)
        for var in all_vars:
            if (hidden_name[0] in var.name) or (hidden_name[1] in var.name):
                var_list.append(var)
        return var_list

    def loss(self, regular_coeffcient=1.0e-2, get_l2_distance=False):
        '''Loss
        '''
        with tf.variable_scope('loss') as scope:
            with tf.variable_scope('l2_distance') as scope:
                self.l2_distance = tf.reduce_mean(tf.pow(self.in_data - self.decoded, 2))

            with tf.variable_scope('l2_regular') as scope:
                pred_fn_pairs = {}
                for i in range(1, 1+self.layer_num):
                    l2_regular_fn = lambda : tf.reduce_mean(tf.pow(self.get_encoded(i), 2))
                    pred_fn_pairs[tf.equal(self.layer_train_ph, i)] = l2_regular_fn
                self.l2_regular = tf.case(pred_fn_pairs, exclusive=True) * regular_coeffcient
            self.loss = self.l2_distance + self.l2_regular

        if get_l2_distance:
            return self.loss, self.l2_distance
        else:
            return self.loss

def main():
    data_shape = [32, 128, 128, 3]
    data = tf.constant(np.random.random(data_shape)-0.5, dtype=tf.float32, shape=data_shape)
    train_ph = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')
    SAE = stack_autoencoder(data, 3, [32, 64, 64], train_ph)
    output = SAE.model()
    layer_train_ph = SAE.get_ph()
    loss, l2_distance = SAE.loss(get_l2_distance=True)
    layer1_var = SAE.get_variable_for_layer(1)
    layer2_var = SAE.get_variable_for_layer(2)
    layer3_var = SAE.get_variable_for_layer(3)
    layers_var = layer1_var + layer2_var + layer3_var
    for var in layers_var:
        print(var.name)

    init = tf.initializers.variables(tf.global_variables())
    with tf.Session() as sess:
        sess.run(init)
        in_data, out_data, distance, loss_ = sess.run([data, output, l2_distance, loss], feed_dict={layer_train_ph:3, train_ph: False})
        print('input:', in_data[0:2, 0:2, 0:2,...])
        print('output:', out_data[0:2, 0:2, 0:2,...])
        print('out_data.shape:', out_data.shape)
        print('l2_distance:', distance)
        print('loss:', loss_)

if __name__ == "__main__":
    import numpy as np
    from casia_webface import casia_webface
    from preprocessing import preprocessing_for_image
    # For display
    from matplotlib import pyplot as plt
    main()

