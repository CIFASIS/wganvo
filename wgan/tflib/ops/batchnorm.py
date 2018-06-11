import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs, is_training=None, update_moving_stats=None, stats_iter=None):
    if axes==[0,2]:
        inputs = tf.expand_dims(inputs, 3)

    # inputs.get_shape()[1] == number of feature maps.
    offset = lib.param(name+'.offset', np.zeros(inputs.get_shape()[1], dtype='float32'))
    scale = lib.param(name+'.scale', np.ones(inputs.get_shape()[1], dtype='float32'))

    moving_mean = lib.param(name+'.moving_mean', np.zeros(inputs.get_shape()[1], dtype='float32'), trainable=False)
    moving_variance = lib.param(name+'.moving_variance', np.ones(inputs.get_shape()[1], dtype='float32'), trainable=False)

    def _batch_norm_training():
        if axes == [0,2,3] or axes == [0,2]:
            # Implementation using 'fused_batch_norm', which is optimized for 4D:
            return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')
        else:
            mean, var = tf.nn.moments(inputs, axes)
            result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
            return result, mean, var

    def _batch_norm_inference():
        if axes == [0,2,3] or axes == [0,2]:
            # Version which blends in the current item's statistics
            #batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            #TODO: temporary fix for when batch_size is 1!
            batch_size = 64
            mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
            mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
            var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
            return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var

            # Standard version
            # return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-2, mean=moving_mean, variance=moving_variance,
            #                               is_training=False, data_format='NCHW')
        else:
            return tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset, scale, 1e-5), moving_mean, moving_variance

    if is_training is None:
        outputs, batch_mean, batch_var = _batch_norm_training()
    else:
        outputs, batch_mean, batch_var = tf.cond(is_training,
                                                   _batch_norm_training,
                                                   _batch_norm_inference)
        if update_moving_stats is not None:
            no_updates = lambda: outputs
            def _force_updates():
                """Internal function forces updates moving_vars if is_training."""
                float_stats_iter = tf.cast(stats_iter, tf.float32)

                update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
                update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))

                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(outputs)
            outputs = tf.cond(update_moving_stats, _force_updates, no_updates)

    if axes == [0,2]:
        return outputs[:,:,:,0] # collapse last dim
    else:
        return outputs
