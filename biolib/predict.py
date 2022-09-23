# %%
import argparse
import subprocess

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    
    with open(args.output, 'w') as f:
        subprocess.call(['rbpnet', 'predict', args.fasta, '--model', args.model, '--output', args.output], stdout=f)
    
    print('Done. You can find the predictions in \'{}\' after downloading the results.'.format(args.output))

# %%
if __name__ == '__main__':
    main()