# %%
import tensorflow as tf
import gin

# %%
@tf.function
def log(x, basis=None):
    if basis is None:
        return tf.math.log(x)
    else:
        return tf.math.log(x) / tf.math.log(tf.cast(basis, x.dtype))

# %%
@tf.function
def kld(q, p, basis=None):
    p = tf.convert_to_tensor(p)
    q = tf.cast(q, p.dtype)
    q = tf.keras.backend.clip(q, tf.keras.backend.epsilon(), 1)
    p = tf.keras.backend.clip(p, tf.keras.backend.epsilon(), 1)
    return tf.reduce_sum(q * log(q / p, basis=basis), axis=-1)


# %%
@gin.register
def logits_to_expected_counts(y, y_pred):
    """Given predicted logits and total counts, return expected counts. 

    This function takes y and y_pred as inputs and computes the total_counts directly from y. 
    This makes it useable as a postproc function, e.g. in custom metrics. 

    Args:
        y      (Tensor): True counts
        y_pred (Tensor): Predicted logits

    Returns:
        Tensor: Expected counts
    """

    # expected shape: batch_size * input_length
    tf.debugging.assert_rank(y, 2)
    tf.debugging.assert_rank(y_pred, 2)

    # sum along sequence axis (to get total counts)
    y_total = tf.math.reduce_sum(y, axis=1)
    y_total = tf.cast(y_total, dtype=tf.float32)
    y_total = tf.expand_dims(y_total, axis=1) # for broadcasting

    # compute softmax of logits along sequence axis
    probs = tf.nn.softmax(y_pred, axis=1)

    # return expected counts, i.e. probs * total_counts
    return probs * y_total
