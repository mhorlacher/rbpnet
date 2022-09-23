# %%
import click
import tensorflow as tf

from rbpnet import io
from rbpnet.models import load_model
from rbpnet.prediction import predict
from rbpnet.metrics import pcc


def evaluate(model, dataset, layer, eval_func):
    print('\t' + '\t'.join(model.tasks))
    for d in dataset:
        name = d[0]['name'][0].numpy().decode('utf-8')
        inputs = d[1]['sequence']
        
        y_true = {key: tf.squeeze(value).numpy() for key, value in d[2].items() if key.endswith(layer)}
        y_pred = {key: tf.squeeze(value).numpy() for key, value in model.predict(inputs).items() if key.endswith(layer)}
        
        print('\t'.join(map(str, [name] + [eval_func(y_true[f'{task}_profile'], y_pred[f'{task}_profile']) for task in model.tasks])))

# %%
@click.command()
@click.argument('tfrecord')
@click.option('-m', '--model')
@click.option('-d', '--dataspec')
@click.option('--layer', default='profile', help='Target layer, i.e. \'profile\' or \'profile_control\'.')
def main(tfrecord, model, dataspec, layer): 
    assert layer in ['profile', 'profile_control']
    
    # load data
    dataspec = io.DataSpec(dataspec)
    data = io.Data([tfrecord], dataspec, use_bias=False)
    dataset = data.dataset(batch_size=1, return_info=True)
    
    # load model
    model = load_model(model)
    model = model.slice_heads([f'{task}_{layer}' for task in model.tasks])
    
    # evaluate
    evaluate(model, dataset, layer, eval_func=pcc)

# %%
if __name__ == '__main__':
    main()