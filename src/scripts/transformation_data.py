from src.data.transformation.transformation_data import DataTransformer
from pathlib import Path

import click
import logging


PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.option("-var", '--variable', default=None, type=click.STRING,
              help="Variable to which to extraction the OpenAQ data")
@click.option("-locations", '--locations_csv_path', type=PATH,
              help="Path to the file where the locations "
                   "of interest are defined")
@click.option("-output", '--output_dir', type=PATH, 
              help="Output directory where to store the data to")
def main(variable, locations_csv_path, output_dir):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    DataTransformer(
        variable,
        locations_csv_path,
        output_dir
    ).run()
