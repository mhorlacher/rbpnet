# %%
from email.policy import default
import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Bio import SeqIO

from rbpnet.utils import sequence2onehot, count_fasta_seqs
from rbpnet.models import load_model

# %%
def explain_step(sequence, model):
    """Returns the 1D attribution for a single input sequence.

    The 2D (input_length, 4) attribution tensor is flattened to a 1D array, as at 
    each position, only the attribution of the observed base is non-zero. 

    Args:
        sequence (str): Input sequence.
        model (Keras Model): RBPNet model.

    Returns:
        np.ndarray: 1D numpy array with attribution values.
    """
    
    return {key: tf.reduce_sum(value, axis=1).numpy() for key, value in model.explain(sequence2onehot(sequence)).items()}

# %%
def explain(fasta, model, output, format):
    n = count_fasta_seqs(fasta)

    with open(output, ('w' if format == 'fasta' else 'wb')) as f, tqdm(total=n) as pbar:
        for seq in SeqIO.parse(fasta, 'fasta'):
            # compute attribution values for sequence
            attribution_dict = explain_step(str(seq.seq), model)
            
            if format == 'fasta':
                attribution_keys = list(attribution_dict.keys())
                # print ids and sequence
                print(f'>{seq.id} ' + ','.join(attribution_keys), file=f)
                print(seq.seq, file=f)
                # print attribution values
                for key in attribution_keys:
                    print(' '.join(map(lambda x: f'{float(x):.3f}', attribution_dict[key])), file=f)
            else:
                # append to numpy file
                np.save(f, {'id': seq.id, 'sequence': seq.seq, 'attributions': attribution_dict}, allow_pickle=True)
                
            # move progress bar
            pbar.update(1)

# %%
@click.command()
@click.argument('fasta')
@click.option('-m', '--model')
@click.option('-o', '--output')
@click.option('--format', default='fasta')
def main(fasta, model, output, format):
    if format not in ['fasta', 'npy']:
        raise ValueError(f'Invalid format: {format}')
    
    # load model
    model = load_model(model)

    # explain
    explain(fasta, model, output, format)


# %%
if __name__ == '__main__':
    main()