# %%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# %%
import gin

# %%
def pcc(a, b):
    return np.corrcoef(a, b)[0, 1]

# %%
@gin.configurable
class PearsonCorrelation(tf.keras.metrics.Mean):
    def __init__(self, post_proc_fn=None, name='pearson_correlation', **kwargs):
        """Pearson Correlation Coefficient. 

        Args:
            post_proc_fn (function, optional): Post-processing function for predicted values. Defaults to None.
        """

        super().__init__(name=name, **kwargs)

        self.post_proc_fn = post_proc_fn
    
    def _compute_correlation(self, y, y_pred):
        # sample_axis = batch-axis
        corr = tfp.stats.correlation(y, y_pred, sample_axis=-1, event_axis=None)
        return corr

    def _nan_to_zero(self, x):
        is_not_nan = tf.math.logical_not(tf.math.is_nan(x))
        is_not_nan = tf.cast(is_not_nan, tf.float32)

        return tf.math.multiply_no_nan(x, is_not_nan)

    def update_state(self, y, y_pred, **kwargs):
        # expected shape: batch_size * input_length
        tf.debugging.assert_rank(y, 2)
        tf.debugging.assert_rank(y_pred, 2)

        if self.post_proc_fn is not None:
            y_pred = self.post_proc_fn(y, y_pred)

        corr = self._compute_correlation(y, y_pred)
        corr = tf.squeeze(corr)

        # remove any nan's that could have been created, e.g. if y or y_pred is a 0-vector
        corr = self._nan_to_zero(corr)

        # assert that there are no inf's or nan's
        tf.debugging.assert_all_finite(corr, f'expected finite tensor, got {corr}')
         
        super().update_state(corr, **kwargs)

# %%
@gin.register
class SpearmanCorrelation(tf.keras.metrics.Mean):
    """Approximation of the Spearman correlation coefficient. 

    Ties are not handled appropriately. Instead of assigning ties equal rank,
    the current implementation will assign unique ranks in arbitrary order to 
    the tied indices. This may generally underestimate the correlation coefficient. 
    """

    def __init__(self, post_proc_fn=None, name='spearman_correlation', **kwargs):
        """Spearman Correlation Coefficient. 

        Args:
            post_proc_fn (function, optional): Post-processing function for predicted values. Defaults to None.
        """

        super().__init__(name=name, **kwargs)

        self.post_proc_fn = post_proc_fn
        if self.post_proc_fn is None:
            self.post_proc_fn = lambda y, y_pred: (y, y_pred)
    
    def _compute_correlation(self, y, y_pred):
        corr = tfp.stats.correlation(y, y_pred, sample_axis=0, event_axis=1)
        return corr

    def _ranks(self, x):
        ranks = tf.argsort(tf.argsort(x, axis=-2), axis=-2)
        return tf.cast(ranks, dtype=tf.float32)

    def update_state(self, y, y_pred, **kwargs):
        y, y_pred = self.post_proc_fn(y, y_pred)

        y_ranks, y_pred_ranks = self._ranks(y), self._ranks(y_pred)
        corr = self._compute_correlation(y_ranks, y_pred_ranks)
        corr = tf.squeeze(corr)
         
        super().update_state(corr, **kwargs)
