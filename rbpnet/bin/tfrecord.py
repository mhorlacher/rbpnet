# %%
from email.policy import default
import click

# %%
import tensorflow as tf
import creme
from tqdm import tqdm

# %%
from rbpnet.io import Sample, DataSpec

# %%
def tfrecord(dataspec, output, window_size=400, shuffle=0):
    dataspec = DataSpec(dataspec)

    # count number of cross-link sites
    n_rows = 0
    for _ in dataspec.sites:
        n_rows += 1
    
    # shuffle
    if shuffle < 0:
        # buffer is set to MAX
        shuffle = n_rows
    elif shuffle == 0:
        site_generator = dataspec.sites
    else:
        # stream-shuffled generator for BED rows (samples)
        site_generator = creme.stream.shuffle(dataspec.sites, buffer_size=shuffle, seed=42)

    with tqdm(total=n_rows) as pbar:
        with tf.io.TFRecordWriter(output) as tfwriter:
            for site in site_generator:
                sample = Sample(site, dataspec, window_size)

                tfwriter.write(sample.serialized)
                pbar.update(1)

# %%
@click.command()
@click.argument('dataspec')
@click.option('-o', '--output')
@click.option('-w', '--window-size', type=int, default=-1)
@click.option('-s', '--shuffle', type=int, default=9)
@click.option('-o', '--output')
def main(dataspec, output, window_size, shuffle):
    tfrecord(dataspec, output, window_size, shuffle)

# %%
if __name__ == '__main__':
    main()