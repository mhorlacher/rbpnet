# %%
import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir')
    parser.add_argument('-t', '--template')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    
    env = Environment(loader=FileSystemLoader('./'))
    template = env.get_template(args.template)
    
    RBP_CELLs = [fpath.name.split('.')[0] for fpath in Path(args.models_dir).glob('*.h5')]
    
    with open(args.output, 'w') as f:
        print(template.render(RBP_CELL=RBP_CELLs), file=f)
    