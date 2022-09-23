# %%
import tensorflow as tf
from tensorflow.keras import layers

# %%
import gin

# %%
@gin.configurable()
class FirstLayerConv(layers.Layer):
    def __init__(self, filters=64, kernel_size=25, use_bias=False, activation='relu', **kwargs):
        super(FirstLayerConv, self).__init__(**kwargs)

        # arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation
        self.kwargs = kwargs

        self.layer = layers.Conv1D(filters, kernel_size, use_bias=use_bias, activation=activation, padding='same')

    def get_config(self):
        config = super(FirstLayerConv, self).get_config()

        # __init__ arguments
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['use_bias'] = self.use_bias
        config['activation'] = self.activation
        config['use_bias'] = self.use_bias
        config.update(self.kwargs)

        return config

    def call(self, x, **kwargs):
        # expected shape: (batch_size, input_length, 4)
        tf.debugging.assert_rank(x, 3)

        x = self.layer(x)
        return x


# %%
@gin.configurable(denylist=['dilation_rate'])
class BodyConv(layers.Layer):
    def __init__(self, filters=64, 
                 kernel_size=3, 
                 dilation_rate=1, 
                 dropout_rate=0.0, 
                 activation='relu', 
                 batch_norm=True, 
                 residual=True,
                 use_bias=True,
                 **kwargs):
        super(BodyConv, self).__init__(**kwargs)

        # arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_bias = use_bias
        self.kwargs = kwargs

        self.conv1d_layer = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, use_bias=use_bias, padding='same')

        if batch_norm:
            self.batch_norm_layer = layers.BatchNormalization()

        self.activation_layer = layers.Activation(activation)

        if dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(dropout_rate)

    def get_config(self):
        config = super(BodyConv, self).get_config()

        # __init__ arguments
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['dilation_rate'] = self.dilation_rate
        config['dropout_rate'] = self.dropout_rate
        config['activation'] = self.activation
        config['batch_norm'] = self.batch_norm
        config['residual'] = self.residual
        config['use_bias'] = self.use_bias
        config.update(self.kwargs)

        return config

    def call(self, inputs, **kwargs):
        # conv
        x = self.conv1d_layer(inputs)

        # batch_norm
        if self.batch_norm:
            x = self.batch_norm_layer(x)

        # activation
        x = self.activation_layer(x)

        # dropout
        if self.dropout_rate > 0.0:
            x = self.dropout_layer(x)

        # residual
        if self.residual:
            x = layers.add([x, inputs])

        return x

# %%
@gin.configurable
class ProfileHead(layers.Layer):
    def __init__(self, kernel_size=25, use_bias=True, **kwargs):
        super(ProfileHead, self).__init__(**kwargs)

        # arguments
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kwargs = kwargs

        self.layer = layers.Conv1DTranspose(1, kernel_size=kernel_size, use_bias=use_bias, padding='same')

    def get_config(self):
        config = super(ProfileHead, self).get_config()

        # __init__ arguments
        config['kernel_size'] = self.kernel_size
        config['use_bias'] = self.use_bias
        config.update(self.kwargs)

        return config

    def call(self, x):
        x = self.layer(x)
        x = tf.squeeze(x, axis=[2]) # 

        # batch_size * input_length
        tf.debugging.assert_rank(x, 2)

        return x

# %%
@gin.configurable
class MultiplicativeTargetBias(layers.Layer):
    def __init__(self, name, **kwargs):
        super(MultiplicativeTargetBias, self).__init__(name=name, **kwargs)
        self.add = layers.Add()

    def call(self, inputs, **kwargs):
        logits_x, logits_y = inputs
        x = self.add([logits_x, logits_y])
        return x

# %%
@gin.configurable(denylist=['name'])
class AdditiveTargetBias(layers.Layer):
    def __init__(self, name, penalty=None, stop_bias_gradient=False, **kwargs):
        super(AdditiveTargetBias, self).__init__(name=name, **kwargs)
        self.kwargs = kwargs
        self.stop_bias_gradient = stop_bias_gradient

        # target/bias distribution similarity penalty
        if penalty is not None:
            self.penalty_fn = penalty(name=f'{name}_{penalty.__name__}')
        else:
            self.penalty_fn = None

    def get_config(self):
        config = super(AdditiveTargetBias, self).get_config()

        # __init__ arguments
        #config['penalty'] = self.penalty
        config['stop_bias_gradient'] = self.stop_bias_gradient
        config.update(self.kwargs)

        return config

    #@tf.function
    def _f(self, x, y):
        # numerical stable version of: log( exp(x) + exp(y) )
        return tf.math.maximum(x, y) + tf.math.log1p( tf.math.exp( -tf.math.abs(x - y) ) )

    def call(self, inputs, training=False, *kwargs):
        logits_t, logits_b, a = inputs

        # stop gradient computation on bias track (i.e. total loss does not influence weight updates of bias head)
        if self.stop_bias_gradient and training:
            logits_b = tf.stop_gradient(logits_b)

        # shape = (batch_size, input_length)
        tf.debugging.assert_rank(logits_t, 2)
        tf.debugging.assert_rank(logits_b, 2)

        # add a penality loss (if specified)
        if self.penalty_fn is not None:
            _ = self.penalty_fn((logits_t, logits_b))

        # Note: r_i = a*p_i + (1−a)*q_i
        log_p = logits_t - tf.expand_dims(tf.math.reduce_logsumexp(logits_t, axis=1), axis=1)
        log_q = logits_b - tf.expand_dims(tf.math.reduce_logsumexp(logits_b, axis=1), axis=1)

        # Note: s_i = log( exp(a + log_p_i) + exp(log_q_i) ) = f(a + log_p, log_q)
        s = self._f(a + log_p, log_q)

        return s

# %%
@gin.configurable(denylist=['name'])
class SequenceAdditiveMixingCoefficient(layers.Layer):
    def __init__(self, name, **kwargs):
        super(SequenceAdditiveMixingCoefficient, self).__init__(name=name, **kwargs)
        
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = self.global_average_pooling(inputs)
        x = self.dense(x)

        tf.debugging.assert_rank(x, 2)

        return x

# %%
@gin.configurable(denylist=['name'])
class ConstantAdditiveMixingCoefficient(layers.Layer):
    def __init__(self, name, **kwargs):
        super(ConstantAdditiveMixingCoefficient, self).__init__(name=name, **kwargs)
        
        self.mixing_coeff = self.add_weight(shape=(), dtype=tf.float32, initializer='random_normal', trainable=True, name=f'{name}_a')
        
    def call(self, *args, **kwargs):
        return self.mixing_coeff
