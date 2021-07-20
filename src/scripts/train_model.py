
from pathlib import Path

import click

from src.constants import ROOT_DIR
from src.models.train_model import ModelTrain

PATH = click.Path(exists=True, path_type=Path)
DATE_TYPE = click.DateTime()

from src.logging import get_logger

logger = get_logger("Model trainer")


# @click.command()
# @click.option('-c', '--config_file', type=PATH, required=True,
#               help="Input file where to take the model configuration from")
def main(
    config_yaml_filename: str,
    config_yaml_parent: Path = ROOT_DIR / "models" / "configuration"
):
    """
    Script to process the CAMS forecasts.
    """

    ModelTrain(config_yaml_filename, config_yaml_parent).run()

    logger.info("Process finished!")


if __name__ == "__main__":
    main("config_inceptiontime_depth6.yml")
