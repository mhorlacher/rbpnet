# %%
from pathlib import Path
import datetime
import shutil

# %%
import gin
import tensorflow as tf

# %%
from rbpnet.models import RBPNet
from rbpnet.io import DataSpec, Data
from rbpnet.utils import soft_subset_dict
from rbpnet.losses import multinomial_loss

# %%
def model_summary_to_file(model, fname):
    """Write a Keras model summary to a file. 

    Args:
        model (tf.keras.model): Keras model
        fname (string): Filename
    """

    with open(fname, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

# %%
def make_callbacks(root_path):
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(root_path), histogram_freq=1)
    csv_cb = tf.keras.callbacks.CSVLogger(str(root_path / 'history.csv'))
    return [tensorboard_cb, csv_cb]

# %%
def create_loss(tasks, loss_fn, use_bias=False, bias_weight=1.0):
    loss_dict, loss_weight_dict = dict(),dict()
    for task in tasks:
        # loss on total track (e.g. eCLIP)
        loss_dict[f'{task}_profile'] = loss_fn
        loss_weight_dict[f'{task}_profile'] = 1.0

        # loss on control track (e.g. SMInput)
        if use_bias:
            loss_dict[f'{task}_profile_control'] = loss_fn
            loss_weight_dict[f'{task}_profile_control'] = bias_weight
    
    return loss_dict, loss_weight_dict

# %%
def create_metrics(tasks, metrics, use_bias=False):
    metrics_dict = dict()
    for task in tasks:
        # profile
        metrics_dict[f'{task}_profile'] = [metric() for metric in metrics]

        # profile bias
        if use_bias:
            metrics_dict[f'{task}_profile_control'] = [metric() for metric in metrics]
    
    return metrics_dict

# %%
@gin.configurable(denylist=['train_data', 'dataspec', 'val_data', 'output'])
def train(
    train_data,
    dataspec,
    config,
    output, 
    val_data=[], 
    model=RBPNet,
    epochs=100,
    batch_size=128,
    optimizer=tf.keras.optimizers.Adam(), 
    cache=True,
    shuffle=None,
    use_bias=False,
    callbacks=[],
    profile_loss=multinomial_loss,
    profile_loss_bias_weight=1.0,
    profile_metrics=[],
):

    # create output root dir (if it doesn't exist)
    output = Path(output) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=True, parents=True)

    # copy 'dataspec.yml' and 'config.gin'
    shutil.copy(dataspec, str(output / 'dataspec.yml'))
    shutil.copy(config, str(output / 'config.gin'))

    # load dataspec
    dataspec = DataSpec(dataspec)

    # create model
    model = model(dataspec)
    model.summary()

    # write model summary
    model_summary_to_file(model, str(output / 'model.summary.txt'))

    # print model inputs/outputs
    print('Model Inputs:\n', model.input_names)
    print('Model Outputs:\n', model.output_names)

    # load train data
    train_data = Data(train_data, dataspec, use_bias)
    train_dataset = train_data.dataset(batch_size=batch_size, shuffle=shuffle, cache=cache)
    train_dataset = train_dataset.map(lambda X, Y: (soft_subset_dict(X, model.input_names), soft_subset_dict(Y, model.output_names)))
    print('Train Dataset:')
    print(train_dataset.element_spec)

    # load val data (optional)
    print('Validation Dataset:')
    if val_data is not None:
        val_data = Data(val_data, dataspec, use_bias)
        val_dataset = val_data.dataset(batch_size=batch_size, shuffle=0, cache=cache)
        val_dataset = val_dataset.map(lambda X, Y: (soft_subset_dict(X, model.input_names), soft_subset_dict(Y, model.output_names)))
        print(val_dataset.element_spec)
    else:
        print('NONE')
        val_dataset = None
    
    # create logging-callbacks and add to optional custom callbacks
    callbacks += (make_callbacks(output) if output is not None else [])
    print('Callbacks:')
    print(callbacks)

    # create metrics and loss
    loss, loss_weight = create_loss(dataspec.tasks, profile_loss, use_bias=use_bias, bias_weight=profile_loss_bias_weight)
    metrics = create_metrics(dataspec.tasks, profile_metrics, use_bias=use_bias)

    # compile model
    model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weight, metrics=metrics)

    # fit model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2, callbacks=callbacks)
    
    # save trained model
    if output is not None:
        model.save(str(output / 'model.h5'))

    # save best validation loss result
    if val_data is not None:
        min_val_loss = str(round(min(history.history['val_loss']), 4))
        with open(str(output.parent / 'result'), 'w') as f:
            print(min_val_loss, end='', file=f)