# %%
import argparse

from rbpnet.io import Fasta

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bed', metavar='<cross-links.bed>')
    parser.add_argument('--fasta')
    parser.add_argument('--input-size', type=int, default=200)
    args = parser.parse_args()

    fasta = Fasta(args.fasta)

    with open(args.bed) as f:
        for line in f:
            chrom, start, end, name, score, strand, *_ = line.strip().split('\t')

            print(f'>{chrom}:{start}-{end}:{strand}:{name}:{score}')
            print(fasta.window(chrom, int(start), strand, args.input_size))

# %%
if __name__ == '__main__':
    main()