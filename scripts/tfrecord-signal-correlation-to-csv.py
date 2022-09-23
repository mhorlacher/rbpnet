# %%
import argparse
from ast import arg
from curses import meta
from json import load

import numpy as np

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy', metavar='<results.npy>')
    parser.add_argument('--tag', default=None)
    parser.add_argument('-o', '--output', metavar='<results.csv>')
    args = parser.parse_args()

    with open(args.output, 'w') as f:
        print('name,sample_name,corr,profile_name', file=f)
        for result_dict in np.load(args.npy, allow_pickle=True).item().values():
            for profile_name, corr in result_dict['corr'].items():
                corr = result_dict['corr'][profile_name]
                row = [result_dict['name'], result_dict['sample_task'], corr, profile_name]
                print(','.join(map(str, row)), file=f)

# %%
if __name__ == '__main__':
    main()