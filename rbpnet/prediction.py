# %%
import numpy as np
import tensorflow as tf

from rbpnet import utils

# %%
def _to_probs(value, key):
    if '_profile' in key:
        value = tf.nn.softmax(value, axis=1)
    elif '_mixing_coefficient':
        value = tf.nn.sigmoid(value)
    else:
        raise ValueError(f'Unknown key: {key}')
    return value

# %%
def predict(inputs, model, to_probs=True):
    pred = model(inputs)
    if to_probs:
        pred = {key: _to_probs(value, key) for key, value in pred.items()}
    return pred

# %%
def predict_from_sequence(sequences, model, **kwargs):
    one_hot = utils.sequences2inputs(sequences)
    pred = predict(one_hot, model, **kwargs)
    if isinstance(sequences, str):
        pred = {key: tf.squeeze(value, axis=0) for key, value in pred.items()}
    return pred

# # %%
# def shuffle_onehot(one_hot):
#     if isinstance(one_hot, tf.Tensor):
#         # tensor to numpy array
#         one_hot = one_hot.numpy()
#     # in-place shuffle along first dimension
#     np.random.shuffle(one_hot)
#     return one_hot

# # %%
# def shuffle_predict(one_hot, model, n=100, **kwargs):
#     shuffled = np.stack([shuffle_onehot(one_hot) for _ in range(n)])
#     return predict(shuffled, model, **kwargs)

# # %%
# def prediction_to_pvalues(pred_observed, pred_null, bonferroni=False):
#     u = pred_null > pred_observed
#     p_values = (tf.math.reduce_sum(tf.cast(u, dtype=tf.int32), axis=0) + 1) / pred_null.shape[0]
    
#     if bonferroni:
#         # Bonferroni correction
#         p_values * pred_null.shape[0]
        
#     return p_values

