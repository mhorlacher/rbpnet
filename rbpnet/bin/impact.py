# %%
import sys

import click

from rbpnet.io import Fasta
from rbpnet.models import load_model


# %% 
def variant_impact(variants, fasta, model, window_size, layer, assert_A_is_ref):
    print('\t' + '\t'.join(model.tasks))
    with open(variants) as f:
        for i, line in enumerate(f):
            chrom, position, strand, base_A, base_B, *rest = line.strip().split('\t')
            
            sequence = fasta.window(chrom, int(position), '+', size=window_size)
            position_in_sequence = int(window_size/2) - 1
            
            if assert_A_is_ref:
                assert sequence[position_in_sequence] == base_A
            
            impact_scores = model.variant_impact(sequence, position_in_sequence, base_A, base_B, reverse_complement=(strand == '-'))
            impact_scores = [f"{impact_scores[f'{task}_{layer}']:.5f}" for task in model.tasks]
            
            rest = ':'.join(rest)
            print(f'{chrom}:{position}:{strand}:{base_A}>{base_B}:{rest}\t' + '\t'.join(impact_scores))
            

# %%
@click.command()
@click.argument('variants')
@click.option('-f', '--fasta')
@click.option('-m', '--model')
@click.option('--assert-a-is-ref', is_flag=True, default=False, help='Assert that allele A is the reference allele.')
@click.option('--layer', default='profile_target', help='Target layer, i.e. \'profile\', \'profile_target\' or \'profile_control\'.')
@click.option('--window-size', default=100)
def main(variants, fasta, model, assert_a_is_ref, window_size, layer):
    assert layer in ['profile', 'profile_target', 'profile_control']
    
    # load model
    model = load_model(model)
    model = model.slice_heads([f'{task}_{layer}' for task in model.tasks])
    
    # load fasta
    fasta = Fasta(fasta)
    
    # compute variant impact
    variant_impact(variants, fasta, model, window_size, layer, assert_a_is_ref)

# %%
if __name__ == '__main__':
    main()