#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:59:12 2020

@author: xuzhang
"""
import tensorflow as tf
from keras import backend as K

def muliti_tangent(target, output, from_logits=False, axis=-1):
    """muliti_tangent between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))

    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of muliti_tangent
        _epsilon = tf.convert_to_tensor(1e-16, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return tf.reduce_sum(tf.tan(tf.pow(tf.abs(target-output),2)), axis)

def Binary_tangent(y_true, y_pred):
    abs_ = K.abs(y_true - y_pred)
    a = K.square(abs_)
    tan = K.sin(a)/K.cos(a)
    return tan