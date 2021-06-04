from pathlib import Path
from datetime import datetime
from typing import Tuple
from src import constants
from src.data.extraction.cams_forecast import CAMSProcessor

import logging
import click
import sys

PATH = click.Path(exists=True, path_type=Path)
DATE_TYPE = click.DateTime()


@click.command()
@click.option('-i', '--input_dir', type=PATH, required=True,
              help="Input directory where to take the data from")
@click.option('-intermediary', '--intermediary_dir', type=PATH, required=True,
              help="Intermediary directory where to store the temporal data")
@click.option('-locations', '--locations_csv_path', type=PATH, help="Path to "
              "the file where the locations of interest are defined", 
              required=True)
@click.option('-o', '--output_dir', type=PATH, required=True,
              help="Output directory where to store the data to")
@click.option('-p', '--time_period', type=click.Tuple([DATE_TYPE, DATE_TYPE]),
              default=None, help="Period of time in which to process the CAMS "
              "data")
def main(
    input_dir: Path, 
    intermediary_dir: Path, 
    locations_csv_path: Path,
    output_dir: Path,
    time_period: Tuple[datetime, datetime]):
    """
    Script to process the CAMS forecasts.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    CAMSProcessor(
        Path(input_dir),
        Path(intermediary_dir),
        Path(locations_csv_path),
        Path(output_dir),
        dict(zip(['start', 'end'], time_period) if time_period else time_period)
    ).run()

    logging.info('Process finished!')
