#! /usr/bin/env python3
# coding: utf-8

'''
Tools from cifar10.
'''

__author__ = 'IriKa'

import re
import tensorflow as tf

def variable_on_cpu(name,
                    shape,
                    initializer,
                    collections=None):
    '''Helper to create a Variable stored on CPU memory.

    Args:
        name:       name of the variable
        shape:      list of ints
        initializer:initializer for Variable
        collections:Add the variable to some collections.

    Returns:
        Variable Tensor
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape,
                              initializer=initializer,
                              dtype=tf.float32,
                              collections=collections)
    return var

def variable_with_weight_decay(name,
                               shape,
                               stddev,
                               wd,
                               collections=None):
    '''Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name:       name of the variable
        shape:      list of ints
        stddev:     standard deviation of a truncated Gaussian
        wd:         add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        collections:Add the variable to some collections.

    Returns:
        Variable Tensor
    '''
    var = variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def activation_summary(x):
    '''Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x:  Tensor
    Returns:
        nothing
    '''
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity',
                                       tf.nn.zero_fraction(x))
