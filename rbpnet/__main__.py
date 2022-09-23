# %%
#import argh
import click

# %%
from rbpnet.bin import train
from rbpnet.bin import predict
from rbpnet.bin import explain
from rbpnet.bin import tfrecord
from rbpnet.bin import impact
from rbpnet.bin import evaluate

# %%
@click.group()
def main():
    pass

# %%
main.add_command(train.main, name='train')
main.add_command(predict.main, name='predict')
main.add_command(explain.main, name='explain')
main.add_command(tfrecord.main, name='tfrecord')
main.add_command(impact.main, name='impact')
main.add_command(evaluate.main, name='evaluate')

# %%
if __name__ == '__main__':
    main()