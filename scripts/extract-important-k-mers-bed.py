# %%
import click
import tqdm
import numpy as np
import tensorflow as tf

from rbpnet.models import load_model
from rbpnet.utils import sequence2onehot
from rbpnet.io import Fasta

# %%
def running_mean(x, k):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[k:] - cumsum[:-k]) / float(k)

# %%
def extract_argmax_kmer(sequence, attribution, k=5):
    i = np.argmax(running_mean(attribution, k=k))
    return int(i), sequence[i:(i+k)], np.sum(attribution[i:(i+k)])

# %%
@click.command()
@click.argument('bed', type=str)
@click.option('-f', '--fasta', type=str)
@click.option('-m', '--model', type=str)
@click.option('-k', '--kmer-size', type=int, default=5)
@click.option('-o', '--output', type=str)
def main(bed, fasta, model, kmer_size, output):
    model = load_model(model)
    fasta = Fasta(fasta)

    with open(output, 'w') as bed_out, open(bed) as bed_in, tqdm.tqdm(total=None) as pbar:
        for line in bed_in:
            row = line.strip().split('\t')
            sequence = fasta.fetch(row[0], int(row[1]), int(row[2]), strand=row[5]).upper()
            onehot = sequence2onehot(sequence)
            attribution = tf.reduce_sum(model.explain(onehot)[f'{model.tasks[0]}_profile_target'], axis=-1)
            assert len(attribution) == len(sequence)
            idx, kmer, score = extract_argmax_kmer(sequence, attribution, k=kmer_size)
            if row[5] == '-':
                idx = len(sequence) - idx - kmer_size
                kmer = kmer[::-1]
            start, end = int(row[1]) + idx, int(row[1]) + idx + kmer_size
            print('\t'.join(map(str, [row[0], start, end, kmer, score, row[5]])), file=bed_out)
            pbar.update(1)


# %%
if __name__ == '__main__':
    main()