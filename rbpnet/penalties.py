# %%
import tensorflow as tf
from tensorflow.keras import layers
import gin

from rbpnet.functions import kld

# %%
class Penalty(layers.Layer):
    """Base class for the target-bias distribution penalty. 

    Layers inheriting from this class compute a similarity between target and bias distribution 
    and add it to the total model loss. This biases the distributions away from each other.
    """

    def __init__(self, weight = 1.0, **kwargs):
        super(Penalty, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.weight = weight

    def get_config(self):
        config = super(Penalty, self).get_config()
        config['weight'] = self.weight
        config.update(self.kwargs)
        return config
    
    @tf.function
    def penalty_fn(self, p_logits, q_logits):
        """Returns the similarity (penality) between two distributions. 

        Args:
            p_logits (Tensor): Tensor of logits of shape (batch_size, sequence_length). 
            q_logits (Tensor): Tensor of logits of shape (batch_size, sequence_length). 

        Raises:
            NotImplementedError: Inheriting classes must overwrite this function. 
        """

        raise NotImplementedError('This is a base class, penalty_fn is not implemented.')

    def call(self, inputs, logits=True, **kwargs):
        p, q = inputs
        if logits:
            p = tf.nn.softmax(p)
            q = tf.nn.softmax(q)
        
        # compute penality
        penalty = self.penalty_fn(p, q)
        
        # sum over batch
        penalty = tf.math.reduce_mean(penalty)
        penalty = penalty * self.weight

        # add penality to loss loss and track as metric
        self.add_loss(penalty)
        self.add_metric(penalty, name=self.name, aggregation="mean")
        
        # return value is not used in most cases
        return penalty

# %%
@gin.configurable(denylist=['name'])
class JSDPenalty(Penalty):
    """Computed the JSD between two distribution and adds it to the total loss.
    """

    def __init__(self, weight = 1.0, **kwargs):
        super(JSDPenalty, self).__init__(weight=weight, **kwargs)
    
    @tf.function
    def penalty_fn(self, p, q):
        m = (p + q) / 2

        return 1 - (kld(p, m, basis=2)/2 + kld(q, m, basis=2)/2)
    
# %%
