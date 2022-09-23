# %%
from tqdm import tqdm
import click
from Bio import SeqIO
import numpy as np

from rbpnet.models import load_model
from rbpnet.utils import count_fasta_seqs

# %%
def predict_to_fasta(fasta, model, output, format): 
    n = count_fasta_seqs(fasta)
    
    with open(output, ('w' if format == 'fasta' else 'wb')) as f, tqdm(total=n) as pbar:
    
        for seq in SeqIO.parse(fasta, 'fasta'):
            predictions = model.predict_from_sequence(str(seq.seq))
            
            if format == 'fasta':
                prediction_keys = list(predictions.keys())
                # print ids and sequence
                print(f'>{seq.id} ' + ','.join(prediction_keys), file=f)
                print(seq.seq, file=f)
                # print attribution values
                for key in prediction_keys:
                    print(' '.join(map(lambda x: f'{float(x):.8f}', predictions[key])), file=f)
            else:
                # append to numpy file
                np.save(f, {'id': seq.id, 'sequence': seq.seq, 'predictions': predictions}, allow_pickle=True)
                
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

    # predict on FASTA sequences
    predict_to_fasta(fasta, model, output, format)


# %%
if __name__ == '__main__':
    main()