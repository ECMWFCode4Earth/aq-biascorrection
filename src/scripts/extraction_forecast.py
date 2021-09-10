from datetime import datetime
from pathlib import Path
from typing import Tuple

import click

from src.data.forecast import CAMSProcessor

PATH = click.Path(exists=True, path_type=Path)
DATE_TYPE = click.DateTime()

from src.logger import get_logger

logger = get_logger("CAMS Forecast Extraction")


@click.command()
@click.option(
    "-i",
    "--input_dir",
    type=PATH,
    required=True,
    help="Input directory where to take the data from",
)
@click.option(
    "-intermediary",
    "--intermediary_dir",
    type=PATH,
    required=True,
    help="Intermediary directory where to store the temporal data",
)
@click.option(
    "-l",
    "--locations_csv_path",
    type=PATH,
    help="Path to " "the file where the locations of interest are defined",
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    type=PATH,
    required=True,
    help="Output directory where to store the data to",
)
@click.option(
    "-p",
    "--time_period",
    type=click.Tuple([DATE_TYPE, DATE_TYPE]),
    default=None,
    help="Period of time in which to process the CAMS " "data",
)
def main(
    input_dir: Path,
    intermediary_dir: Path,
    locations_csv_path: Path,
    output_dir: Path,
    time_period: Tuple[datetime, datetime],
):
    """
    Script to process the CAMS forecasts.
    """
    CAMSProcessor(
        Path(input_dir),
        Path(intermediary_dir),
        Path(locations_csv_path),
        Path(output_dir),
        dict(zip(["start", "end"], time_period) if time_period else time_period),
    ).run()

    logger.info("Process finished!")
