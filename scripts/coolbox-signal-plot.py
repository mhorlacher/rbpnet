# %%
import argparse
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import coolbox.api as cb

from rbpnet.io import Data, DataSpec
from rbpnet.utils import load_model

# %%
def predict(sequence_onehot, model, target):
    return tf.math.softmax(tf.squeeze(model(sequence_onehot)[target])).numpy()

# %%
def values_to_bedGraph(fname, values, chrom, start):
    with open(fname, 'w') as f:
        for i, v in enumerate(values, start=start):
            print(f'{chrom}\t{i}\t{i+1}\t{v}', file=f)

# %%
def plot_coolbox(name, y_true, y_pred):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        y_true_file = str(tmpdir / 'y_true.bedGraph')
        y_pred_file = str(tmpdir / 'y_pred.bedGraph')

        #print(y_true)
        #print(y_pred)

        values_to_bedGraph(y_true_file, y_true, chrom='chrom_xyz', start=0)
        values_to_bedGraph(y_pred_file, y_pred, chrom='chrom_xyz', start=0)

        RANGE = f'chrom_xyz:0-{y_pred.shape[0]}'

        frame = cb.XAxis(name=name, title=name)
        frame += cb.BedGraph(y_true_file, color='blue', min_value=0.0, max_value=float(max(y_true))) + cb.TrackHeight(3.5) + cb.Title('True Counts')
        frame += cb.BedGraph(y_pred_file, color='red', min_value=0.0) + cb.TrackHeight(3.5) + cb.Title('Pred Signal')
        frame += cb.Spacer(1.0)
        #frame += cb.BedGraph('attributions.bedGraph', color='green') + cb.TrackHeight(2) + cb.Title('RBPNet IG Info')
        #frame += cb.BedGraph('SARS-CoV-2.predictions.bedGraph', color='blue') + cb.TrackHeight(3.5) + cb.Title('Pysster')
        #frame += cb.BigWig('test.bw', color = 'red') + cb.Title('RBPNet')
        #frame += Spacer(1.0)

        print(name)
        fig = frame.plot(RANGE)

        return fig

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecords', nargs='+')
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dataspec')
    parser.add_argument('--target')
    parser.add_argument('-o', '--output', default='figures/{name}.png')
    args = parser.parse_args()

    model = load_model(args.model)

    dataspec = DataSpec(args.dataspec)
    data = Data(args.tfrecords, dataspec)
    dataset = data.dataset(batch_size=1, shuffle=10_000, cache=False, return_info=True)

    for sample in dataset:
        name = sample[0]['name'].numpy()[0].decode('UTF-8')
        y_pred = predict(sample[1]['sequence'], model, args.target)
        y_true = sample[2][args.target]
        fig = plot_coolbox(name, y_true[0].numpy(), y_pred)

        fig.savefig(args.output.format(name=name.replace('|', '_').replace(':', '_')))

# %%
if __name__ == '__main__':
    main()