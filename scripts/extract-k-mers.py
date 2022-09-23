# %%
import click
import numpy as np

# %%
def running_mean(x, k):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[k:] - cumsum[:-k]) / float(k)

# %%
def load_attributions(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith('>'):
                name = line.strip().split(' ')[0][1:]
            sequence = f.readline().strip()
            attribution = list(map(float, f.readline().strip().split(' ')))
            yield name, sequence, attribution

# %%
def extract_argmax_kmer(sequence, attribution, k=5):
    i = np.argmax(running_mean(attribution, k=k))
    return sequence[i:(i+k)], np.sum(attribution[i:(i+k)])

# %%
@click.command()
@click.argument('fasta-ig')
@click.option('-k', '--kmer-size', type=int, default=5)
def main(fasta_ig, kmer_size):
    print('\tkmer\tscore')
    for name, sequence, attribution in load_attributions(fasta_ig):
        kmer, score = extract_argmax_kmer(sequence, attribution, k=kmer_size)
        print(f'{name}\t{kmer}\t{score:.4f}')

# %%
if __name__ == '__main__':
    main()