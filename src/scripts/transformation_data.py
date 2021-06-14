import logging
import sys
from pathlib import Path

import click

from src import constants
from src.data.transformation.transformation_data import DataTransformer

PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument("var", '--variable', type=click.Choice(['pm25', 'no2', 'o3', 'all']))
@click.option("-l", '--locations_csv_path', type=PATH,
              help="Path to the file where the locations "
                   "of interest are defined")
@click.option("-o", '--output_dir', type=PATH, 
              help="Output directory where to store the data to")
def main(variable: str, locations_csv_path: Path, output_dir: Path):
    """ Transform data.

    VAR is the variable to transform.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    DataTransformer(
        variable,
        locations_csv_path,
        output_dir
    ).run()
    logging.info('Process finished!')
