# %%
import tensorflow as tf
import igrads

# %%
def attribution(inputs, model, atype='IG', steps=50):
    """Compute sequence attribution(s) for a given model and inputs.

    Args:
        inputs (tf.Tensor or np.ndarray): 2D (input_length, 4) tensor of onehot-encoded sequence.
        model (Keras Model): Model to compute attribution for.

    Returns:
        tf.Tensor: Feature attributions.
    """
    
    pred = model.predict(tf.expand_dims(inputs, axis=0))
    if atype == 'atype':
        return igrads.integrated_gradients(inputs, model, target_mask=pred, steps=steps)
    elif atype == 'grad_x_input':
        return igrads.grad_x_input(inputs, model, target_mask=pred)
    else:
        raise ValueError(f'Unrecognized attribution type {atype}.')