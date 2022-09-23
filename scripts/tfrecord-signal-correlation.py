"""Compute the pairwise signal correlation coefficient between samples of two tfrecords. 
"""

# %%
import argparse
import copy
from multiprocessing.sharedctypes import Value

import tqdm
import numpy as np
import tensorflow as tf

from rbpnet.io import DataSpec, Data

# %%
def load_dataset(tfrecords, dataspec, use_bias=False):
    data = Data(tfrecords, dataspec, use_bias=use_bias)
    return data.dataset(batch_size=1, return_info=True)

# %%
def get_task_from_name(name):
    return '_'.join(name.split(':')[-1].split('_')[:-1])

# %%
def pcc(a, b):
    return np.corrcoef(a, b)[0, 1]

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord-1', nargs='+')
    parser.add_argument('--tfrecord-2', nargs='+')
    parser.add_argument('--dataspec-1')
    parser.add_argument('--dataspec-2')
    parser.add_argument('--use-bias', action='store_true', default=False)
    parser.add_argument('--tag-1', default='1')
    parser.add_argument('--tag-2', default='2')
    parser.add_argument('-o', '--output', default='result.npy')
    args = parser.parse_args()

    # load dataspecs
    dataspec_1 = DataSpec(args.dataspec_1)
    dataspec_2 = DataSpec(args.dataspec_2)

    # load datasets
    dataset_1 = load_dataset(args.tfrecord_1, dataspec_1, use_bias=args.use_bias)
    dataset_2 = load_dataset(args.tfrecord_2, dataspec_2, use_bias=args.use_bias)

    n = 0
    for (x, y) in zip(dataset_1, dataset_2):
        n += 1

    results = {}
    with tqdm.tqdm(total=n) as pbar:
        for (sample_1, sample_2) in zip(dataset_1, dataset_2):
            # assert that datasets are sorted
            if sample_1[0]['name'][0] != sample_2[0]['name'][0]:
                raise ValueError(f"TFRecords need to be sorted! ({sample_1[0]['name'][0]} != {sample_2[0]['name'][0]})")

            result = {}
            
            result['name'] = sample_1[0]['name'][0].numpy().decode('UTF-8')
            result['sample_task'] = get_task_from_name(result['name'])
            
            result[f'counts_{args.tag_1}'] = {k: tf.squeeze(v).numpy() for k, v in sample_1[2].items()}
            result[f'counts_{args.tag_2}'] = {k: tf.squeeze(v).numpy() for k, v in sample_2[2].items()}
            
            result[f'countsTotal_{args.tag_1}'] = {k: tf.reduce_sum(v).numpy() for k, v in sample_1[2].items()}
            result[f'countsTotal_{args.tag_2}'] = {k: tf.reduce_sum(v).numpy() for k, v in sample_2[2].items()}
            
            # compute correlation
            result['corr'] = {k: pcc(result[f'counts_{args.tag_1}'][k], result[f'counts_{args.tag_2}'][k]) for k in result['counts_1']}
            
            results[result['name']] = copy.deepcopy(result)
            pbar.update(1)

    # save evaluation dict
    np.save(allow_pickle=True, arr=results, file=args.output)

# %%
if __name__ == '__main__':
    main()