# %%
import tensorflow as tf

from rbpnet import CUSTOM_OBJECTS
from rbpnet.prediction import predict, predict_from_sequence
from rbpnet.attribution import attribution
from rbpnet.variant_impact import variant_impact
from rbpnet.utils import slice_model, get_model_tasks, get_model_use_bias

# %%
def __predict(self, inputs, **kwargs):
    """Returns model predictions on inputs with logits to probs.
    """
    
    return predict(inputs, model=self, **kwargs)

def __predict_from_sequence(self, sequences, **kwargs):
    """Predicts on RNA/DNA sequences.

    Args:
        sequences (str, list): RNA/DNA sequence(s). If a string, it is assumed to be a single sequence.

    Returns:
        dict: Dictionary of predictions.
    """
    
    return predict_from_sequence(sequences, model=self, **kwargs)

def __explain(self, inputs, **kwargs):
    return attribution(inputs, self, **kwargs)

def __variant_impact(self, sequence, position, base_A, base_B, **kwargs):
    return variant_impact(self, sequence, position, base_A, base_B, **kwargs)

def __slice_heads(self, output_names):
    sliced_model = slice_model(self, self.input_names, output_names)
    __add_attributes_and_bound_methods(sliced_model)
    
    # add attributes
    sliced_model.tasks = self.tasks
    sliced_model.use_bias = self.use_bias
    
    return sliced_model

def __add_attributes_and_bound_methods(model):
    # add custom bound methods (this is a workaround for keras's buggy model sub-classing)
    model.predict = __predict.__get__(model)
    model.predict_from_sequence = __predict_from_sequence.__get__(model)
    model.explain = __explain.__get__(model)
    model.variant_impact = __variant_impact.__get__(model)
    model.slice_heads = __slice_heads.__get__(model)
    
    # add attributes
    model.tasks = get_model_tasks(model)
    model.use_bias = get_model_use_bias(model)

# %%
def load_model(filepath, compile=False, **kwargs):
    # load model.h5
    model = tf.keras.models.load_model(filepath,compile=compile, custom_objects=CUSTOM_OBJECTS, **kwargs)
    __add_attributes_and_bound_methods(model)
    return model
