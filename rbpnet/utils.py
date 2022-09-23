# %%
from pathlib import Path
import math

# %%
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd

# %%
from rbpnet import CUSTOM_OBJECTS
from rbpnet import functions, losses, layers

# %%
def count_fasta_seqs(fasta):
    n = 0
    with open(fasta) as f:
        for line in f:
            if line[0] == '>':
                n += 1
    return n

# %%
def predict(model, x):
    output = model.predict(x[0])
    output_dict = {name: pred for name, pred in zip(model.output_names, output)}
    return output_dict

# %%
def evaluate(inputs, model):
    loss = model.evaluate(tf.data.Dataset.from_tensor_slices(inputs).take(1).batch(1))
    return {name: l for name, l in zip(['loss'] + model.output_names, loss)}

# %%
@tf.function
def subset_dict(d, keys):
    return dict(map(lambda x: (x, d.get(x, None)), keys))

# %%
@tf.function
def soft_subset_dict(d, keys):
    keys = [key for key in d]
    return dict(map(lambda x: (x, d.get(x, None)), keys))

# %%
def nan_to_zero(x):
    """Replaces nan's with zeros."""
    return 0 if math.isnan(x) else x

# %%
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# %%
def sequence2int(sequence):
    return [base2int.get(base, 999) for base in sequence]

# %%
def sequence2onehot(sequence):
    return tf.one_hot(sequence2int(sequence), depth=4)

# %%
def sequences2inputs(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]
    return tf.one_hot([sequence2int(s) for s in sequences], depth=4)

# %%

# %%
baseComplement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# %%
def reverse_complement(dna_string):
    """Returns the reverse-complement for a DNA string."""

    complement = [baseComplement.get(base, 'N') for base in dna_string]
    reversed_complement = reversed(complement)
    return ''.join(list(reversed_complement))

# %%
def read_beds(beds):
    for bed in beds:
        with open(bed) as f:
            for line in f:
                row = line.strip().split('\t')
                yield row


# %%
# def load_model(filepath, compile=False, **kwargs):
#     m = tf.keras.models.load_model(filepath,compile=compile, custom_objects=CUSTOM_OBJECTS, **kwargs)
#     return m

# %%
@tf.function
def subset_dict(d, keys):
    return dict(map(lambda x: (x, d.get(x, None)), keys))

# %%
def parse_yaml_to_dict(yaml_file):
    with open(yaml_file) as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

# %%
def predict_to_dict(X, model):
    """Return model predictions y_pred as a dictionary. 

    Args:
        X (Tensor): Input tensor
        model (keras.Model): Keras Model

    Returns:
        dict: Dictionary mapping output layer names to output tensors
    """

    y_pred = model(X)
    return {name: pred for name, pred in zip(model.output_names, y_pred)}

# %%
def calc_profile_loss_to_dict(y_dict, logits_dict, tasks):
    profile_loss_dict = {}
    for task in tasks:
        loss = losses.multinomial_loss(y_dict[f'{task}_profile'], logits_dict[f'{task}_profile'])
        loss = tf.reshape(loss, shape=(1,))
        profile_loss_dict[f'{task}_profile_loss'] = loss
    return profile_loss_dict

# %%
def calc_expected_counts_to_dict(y_dict, logits_dict, tasks):
    expected_counts_dict = {}
    for task in tasks:
        y, y_pred = y_dict[f'{task}_profile'], logits_dict[f'{task}_profile']
        expected_counts = functions.logits_to_expected_counts(y, y_pred)
        expected_counts_dict[f'{task}_expected_counts'] = expected_counts
    return expected_counts_dict

# %%
def format_y_y_pred_to_df(y, y_pred, task):
    """Sequeezes two tensors to 1D, then creates a pandas dataframe. 

    Make sure that there is no (or an empty) batch-dimension!   

    Args:
        y (Tensor): Truth
        y_pred ([Tensor): Predicted
        task (string): Task, e.g. 'RBL_CELL'

    Returns:
        pandas.DataFrame: Pandas dataframe containing columns [y, y_pred, task]
    """

    y = tf.squeeze(y)
    y_pred = tf.squeeze(y_pred)

    df = pd.DataFrame(np.stack([y, y_pred]).transpose(), columns=['y', 'y_pred'])
    df['task'] = task
        
    return df

# %%
def slice_model(model, input_layer_names, output_layer_names):
    input_layers = {name: model.get_layer(name).input for name in input_layer_names}
    output_layers = {name: model.get_layer(name).output for name in output_layer_names}

    return tf.keras.models.Model(inputs=input_layers, outputs=output_layers)

# %%
def slice_model_nodict(model, input_layer_names, output_layer_names):
    input_layers = [model.get_layer(name).input for name in input_layer_names]
    output_layers = [model.get_layer(name).output for name in output_layer_names]

    return tf.keras.models.Model(inputs=input_layers, outputs=output_layers)

# %%
def get_model_tasks(model):
    return [name[:-8] for name in model.output_names if name.endswith('_profile')]

# %%
def get_model_use_bias(model):
    for layer in model.layers:
        if isinstance(layer, layers.AdditiveTargetBias):
            return True
    else:
        return False
    