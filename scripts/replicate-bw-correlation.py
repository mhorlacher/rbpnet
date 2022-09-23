# %%
import argparse

from rbpnet.io import Track
from rbpnet.metrics import pcc

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('crosslink_bed')
    parser.add_argument('--rep-1-bw-pos')
    parser.add_argument('--rep-1-bw-neg')
    parser.add_argument('--rep-2-bw-pos')
    parser.add_argument('--rep-2-bw-neg')
    parser.add_argument('--window-size', type=int, default=200)
    args = parser.parse_args()

    track_1 = Track(args.rep_1_bw_pos, args.rep_1_bw_neg)
    track_2 = Track(args.rep_2_bw_pos, args.rep_2_bw_neg)
    
    with open(args.crosslink_bed) as f:
        for line in f:
            chrom, start, end, name, score, strand, *_ = line.strip().split('\t')
            name = f'{chrom}:{start}-{end}:{strand}:{name}'
            
            profile_1 = track_1.window(chrom, int(start), strand, size=args.window_size)
            profile_2 = track_2.window(chrom, int(start), strand, size=args.window_size)
            
            print(f'{name}\t{pcc(profile_1, profile_2)}\t{sum(profile_1)}\t{sum(profile_2)}')

# %%
if __name__ == '__main__':
    main()