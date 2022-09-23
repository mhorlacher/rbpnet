# %%
import tensorflow as tf
from tensorflow.keras import layers
import gin

# %%
from rbpnet import layers as rlayers
from rbpnet.load import load_model # legacy support

# %%
def _make_output_head(x, task, use_bias, target_bias_layer, mixing_coeff_layer):
    px_t = rlayers.ProfileHead(name=f'{task}_profile' + ('_target' if use_bias else ''))(x)

    if use_bias:
        px_b = rlayers.ProfileHead(name=f'{task}_profile_control')(x)
        a = mixing_coeff_layer(name=f'{task}_mixing_coefficient')(x)
        px = target_bias_layer(name=f'{task}_profile')((px_t, px_b, a))
    else:
        px = px_t
        px_b, px_t, a = None, None, None

    return px, px_b, px_t, a

# %%
@gin.configurable(denylist=['dataspec'])
def RBPNet(
    dataspec, 
    conv_layers=9, 
    dilation=True, 
    use_bias=True, 
    target_bias_layer=rlayers.AdditiveTargetBias, 
    mixing_coeff_layer=rlayers.SequenceAdditiveMixingCoefficient):

    # input
    x_in = layers.Input(shape=(None, 4), name='sequence')

    # body
    x = rlayers.FirstLayerConv()(x_in)
    for i in range(1, conv_layers+1):
        dilation_rate = i**2 if dilation else 1
        x = rlayers.BodyConv(dilation_rate=dilation_rate, name=f'body_conv_{i}')(x)
    
    # output
    x_out = dict()
    for task in dataspec.tasks:
        px, px_b, px_t, a = _make_output_head(x, task, use_bias, target_bias_layer, mixing_coeff_layer)

        x_out[f'{task}_profile'] = px
        if px_b is not None:
            x_out[f'{task}_profile_control'] = px_b
            x_out[f'{task}_profile_target'] = px_t
            x_out[f'{task}_mixing_coefficient'] = a
    
    model = tf.keras.Model(inputs=x_in, outputs=x_out, name='RBPNet')
    return model
