# %%
import argparse

import numpy as np
import tensorflow as tf

from rbpnet.utils import sequence2onehot

# %%
def dist(motif, reference_motif):
    return np.sum(motif * reference_motif)

# %%
def max_dist(a, b):
    padding = int(min([a.shape[0], b.shape[0]])/2)
    a_pad = tf.pad(a, [[padding, padding], [0, 0]])

    dists = []
    for i in range(a_pad.shape[0] - b.shape[0] + 1):
        a_pad_loc = a_pad[i:(i+5)]
        dists.append(tf.reduce_sum(a_pad_loc * b))
    return max(dists).numpy()

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kmers', metavar='<kmers.tsv>')
    parser.add_argument('-m', '--motif')
    args = parser.parse_args()

    reference_motif = sequence2onehot(args.motif)
    with open(args.kmers) as f:
        _ = f.readline()
        for line in f:
            motif_id, motif, score = line.strip().split('\t')
            motif = sequence2onehot(motif)
            print(motif_id, max_dist(motif, reference_motif))

# %%
if __name__ == '__main__':
    main()