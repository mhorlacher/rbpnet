# %%
import click
import gin

# %%
from rbpnet.train import train

# %%
@click.command()
@click.argument('train_data', nargs=-1)
@click.option('--validation-data', default=None)
@click.option('-c', '--config', default=None)
@click.option('-d', '--dataspec', default=None)
@click.option('-o', '--output', default=None)
def main(train_data, validation_data, config, dataspec, output):
    """Train an RBPNet model. """

    # parse configs (train, model, etc.) if specified
    if config is not None:
        gin.parse_config_file(config)

    # launch training
    train(list(train_data), dataspec, config, output, val_data=validation_data)

# %%
if __name__ == '__main__':
    main()