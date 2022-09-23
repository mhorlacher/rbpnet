# %%
import argparse
import tempfile
import subprocess
from pathlib import Path

# %%
import papermill as pm
from nbconvert import HTMLExporter

# %%
#from rbpnet.notebooks import NOTEBOOK_EVAL

# %%
def parse_args():
    """Parse CLI arguments. 

    Returns:
        argparse.Namespace: Arguments namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='eval') # detect format by suffix

    # essential args
    parser.add_argument('--model-h5')
    parser.add_argument('--dataspec-yml')
    parser.add_argument('--tfrecords', nargs='+')
    parser.add_argument('--use-bias', action='store_true', default=False) # hopefully we can parse this directly from the config in the future..

    # optional args
    parser.add_argument('--eval-notebook', default=None)

    return parser.parse_args()

# %%
def run_pmill(eval_notebook, output, **kwargs):
    """Runs papermill and jupyter nbconvert. 

    First, papermill injects parameters and executes the notebook. 
    Then, jupyter nbconvert convert the notebook to the desired format. 

    Args:
        eval_notebook (string): Path to the evaluation notebook
        output (string): Output path
    """

    kwargs.update({'output': output})
    Path(output).mkdir(parents=False, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_notebook = str(Path(tmpdir) / 'prepared-notebook.ipynb')
        
        print(kwargs)

        pm.execute_notebook(
            eval_notebook,
            prepared_notebook,
            parameters = kwargs,
            prepare_only = False
        )

        cmd = f'jupyter nbconvert --to html --output-dir {output} --output eval {prepared_notebook}'
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True)


# %%
def main():
    args = parse_args()
    run_pmill(**vars(args))

# %%
if __name__ == '__main__':
    main()



