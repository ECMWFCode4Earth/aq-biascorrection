import logging
from pathlib import Path

import click

from src.models.train_model import ModelTrain

PATH = click.Path(exists=True, path_type=Path)
DATE_TYPE = click.DateTime()

logger = logging.getLogger("Model trainer")


# @click.command()
# @click.option('-c', '--config_file', type=PATH, required=True,
#               help="Input file where to take the model configuration from")
def main(
        config_yaml_path,
):
    """
    Script to process the CAMS forecasts.
    """

    ModelTrain(
        config_yaml_path.name,
        config_yaml_path.parent
    ).run()

    logger.info('Process finished!')


if __name__ == '__main__':
    main(Path('/home/pereza/git/esowc/aq-biascorrection/models/configuration/'
              'config_inceptiontime_depth6.yml'))