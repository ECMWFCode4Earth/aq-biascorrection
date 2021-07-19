import logging
import sys
from pathlib import Path

import click
import pandas as pd

from src import constants
from src.data.openaq_obs import OpenAQDownloader
from src.data.utils import Location

PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument("var", type=click.Choice(["pm25", "no2", "o3", "all"]))
@click.option(
    "-l",
    "--locations_csv_path",
    type=PATH,
    default=constants.ROOT_DIR / "data/external/stations.csv",
    help="Path to the file where the locations "
    "of interest are defined in .csv format",
)
@click.option(
    "-o",
    "--output_dir",
    type=PATH,
    default=constants.ROOT_DIR / "data/interim/observations",
    help="Output directory where to store the data to",
)
def main(
    var: str,
    locations_csv_path: Path = constants.ROOT_DIR / "data/external/stations.csv",
    output_dir: Path = constants.ROOT_DIR / "data/interim/observations",
):
    """
    This function reads a csv file with the following structure:
    id,city,country,latitude,longitude,timezone

    AE001,Dubai,United Arab Emirates,25.0657,55.17128,Asia/Dubai

    ............................................................
    ............................................................
    ............................................................

    For each of the rows of the csv file, it runs the OpenAQDownloader
    for the respective variable / location combination and  stores the
    data at the output directory (given as an argument)

    VAR is the variable to extract from OpenAQ API.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    logger = logging.getLogger("OpenAQ download Pipeline")

    varnames = ["pm25", "no2", "o3"] if var == "all" else [var]
    for variable in varnames:
        locations_df = pd.read_csv(locations_csv_path)
        number_of_successful_locations = 0
        for location in locations_df.iterrows():
            loc = Location(
                location[1]["id"],
                location[1]["city"],
                location[1]["country"],
                location[1]["latitude"],
                location[1]["longitude"],
                location[1]["timezone"],
                location[1]["elevation"],
            )
            logger.info(f"Starting process for location of interest {str(loc)}")
            downloader = OpenAQDownloader(loc, output_dir, variable)
            try:
                output_path = downloader.run()
                number_of_successful_locations += 1
            except Exception as ex:
                logger.error(str(ex))
                continue
        logger.info(
            f"The number of locations which has been correctly "
            f"downloaded is {number_of_successful_locations} out of"
            f" {len(locations_df)} for variable {variable}"
        )

    logger.info("Process finished!")
