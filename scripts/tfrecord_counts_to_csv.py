# %%
import argparse

# %%
import pandas as pd
import tensorflow as tf

# %%
from rbpnet import io
from rbpnet import utils

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataspec')
    parser.add_argument('--tfrecords', nargs='+')
    parser.add_argument('-o', '--output', default='counts.csv')
    args = parser.parse_args()

    # load dataspec
    dataspec_dict = utils.parse_yaml_to_dict(args.dataspec)
    tasks = list(dataspec_dict['task_specs'].keys())

    # load dataset
    dataset = io.make_tfrecord_dataset(args.tfrecords, args.dataspec, batch_size=None, shuffle=None, cache=False, use_bias=True, return_info=True)

    task_counts = {task: {'name': [], 'origin_task': [], 'count': [], 'count_control': []} for task in tasks}
    for sample in dataset:
        for task in tasks:
            name = sample[0]['name'].numpy().decode('UTF-8')
            count = tf.math.reduce_sum(sample[2][f'{task}_profile']).numpy()
            count_control = tf.math.reduce_sum(sample[2][f'{task}_profile_control']).numpy()

            task_counts[task]['name'] += [name]
            task_counts[task]['origin_task'] += ['_'.join(name.split(':')[-1].split('_')[:-1])]
            task_counts[task]['count'] += [count]
            task_counts[task]['count_control'] += [count_control]

    task_df_list = []
    for task in tasks:
        df = pd.DataFrame(task_counts[task])
        df['task'] = task
        task_df_list.append(df)
    task_df = pd.concat(task_df_list, axis=0)
    task_df.to_csv(args.output)