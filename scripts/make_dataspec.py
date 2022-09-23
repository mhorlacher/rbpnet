# %%
import argparse

# %%
from snakemake.io import glob_wildcards

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-bw-pos', metavar='<{task}/signal.pos.bw>')
    parser.add_argument('--signal-bw-neg', metavar='<{task}/signal.neg.bw>')
    parser.add_argument('--signal-control-bw-pos', metavar='<{task}/signal.control.pos.bw>')
    parser.add_argument('--signal-control-bw-neg', metavar='<{task}/signal.control.neg.bw>')
    parser.add_argument('--peaks', metavar='<{task}/peaks.crosslink.bed>')
    parser.add_argument('--fasta', metavar='<genome.fasta>')
    args = parser.parse_args()

    tasks = glob_wildcards(args.peaks).task

    # write dataspec
    print('fasta_file:', args.fasta)
    print('task_specs:\n')
    for task in tasks:
        print(' '*2 + f'{task}:')

        print(' '*4 + 'tracks:')
        print(' '*6 + '- ' + args.signal_bw_pos.format(task=task))
        print(' '*6 + '- ' + args.signal_bw_neg.format(task=task))

        print(' '*4 + 'control:')
        print(' '*6 + '- ' + args.signal_control_bw_pos.format(task=task))
        print(' '*6 + '- ' + args.signal_control_bw_neg.format(task=task))

        print(' '*4 + 'peaks:', args.peaks.format(task=task))
        print()