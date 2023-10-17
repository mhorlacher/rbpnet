# %%
__version__ = '0.10.0'

# %%
# Disable tensorflow INFO and WARNING log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
import gin
import gin.tf

import tensorflow as tf

# %%
# set dynamic memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# %%
from rbpnet import layers
from rbpnet import losses
from rbpnet import metrics
from rbpnet import penalties
from rbpnet import models
from rbpnet import utils

# %%
# Custom layers
CUSTOM_LAYERS = {
    'FirstLayerConv': layers.FirstLayerConv,
    'BodyConv': layers.BodyConv,
    'ProfileHead': layers.ProfileHead,
    'MultiplicativeTargetBias': layers.MultiplicativeTargetBias,
    'AdditiveTargetBias': layers.AdditiveTargetBias,
    'SequenceAdditiveMixingCoefficient': layers.SequenceAdditiveMixingCoefficient,
    'ConstantAdditiveMixingCoefficient': layers.ConstantAdditiveMixingCoefficient,
    'JSDPenalty': penalties.JSDPenalty,
    }

# Custom losses
CUSTOM_LOSSES = {
    'multinomial_loss': losses.multinomial_loss
    }

# Custom metrics
CUSTOM_METRICS = {
    'PearsonCorrelation': metrics.PearsonCorrelation,
    'SpearmanCorrelation': metrics.SpearmanCorrelation
    }

# Custom objects
CUSTOM_OBJECTS = {**CUSTOM_LAYERS, **CUSTOM_LOSSES, **CUSTOM_METRICS}

# %%
# Configure gin externals
gin.config.external_configurable(tf.keras.callbacks.EarlyStopping, module='tf.keras.callbacks')
gin.config.external_configurable(tf.keras.losses.mae, module='tf.keras.losses')
gin.config.external_configurable(tf.keras.optimizers.Adam, module='tf.keras.optimizers')
