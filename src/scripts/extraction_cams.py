from pathlib import Path
from src.data.extraction.cams_forecast import CAMSProcessor

import logging
import click


FILE = click.Path(exists=True, path_type=Path)

@click.command()
@click.option('-i', '--input_dir', type=FILE, required=True,
              help="Input directory where to take the data from")
@click.option('-intermediary', '--intermediary_dir', type=FILE, required=True,
              help="Intermediary directory where to store the temporal data")
@click.option('-locations', '--locations_csv_path', type=FILE, help="Path to "
              "the file where the locations of interest are defined", 
              required=True)
@click.option('-o', '--output_dir', type=FILE, required=True,
              help="Output directory where to store the data to")
@click.option('-p', '--time_period', default=None, help="Period of time in "
              "which to process the CAMS data")
def main(input_dir, intermediary_dir, locations_csv_path, output_dir, time_period):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    CAMSProcessor(
        Path(input_dir),
        Path(intermediary_dir),
        Path(locations_csv_path),
        Path(output_dir),
        time_period
    ).run()

    logging.info('Process finished!')
