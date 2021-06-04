from pathlib import Path
from src import constants
from src.data.extraction.openaq_obs import OpenAQDownloader
from src.data.utils import Location

import pandas as pd
import logging
import click
import sys


PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.option("-locations", '--locations_csv_path', type=PATH, required=True,
              help="Path to the file where the locations of interest are defined")
@click.option("-output", '--output_dir', type=PATH, required=True,
              help="Output directory where to store the data to")
@click.option("-var", '--variable', default=None, type=str, 
              help="Variable to which to extraction the OpenAQ data")
def download_openaq_data_from_csv_with_locations_info(
        csv_path: Path,
        output_dir: Path,
        variable: str
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
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    logger = logging.getLogger("OpenAQ download Pipeline")
    locations_df = pd.read_csv(csv_path)
    number_of_successful_locations = 0
    for location in locations_df.iterrows():
        loc = Location(
            location[1]['id'],
            location[1]['city'],
            location[1]['country'],
            location[1]['latitude'],
            location[1]['longitude'],
            location[1]['timezone'],
            location[1]['elevation']
        )
        logger.info(f"Starting process for location of"
                     f" interest {str(loc)}")
        downloader = OpenAQDownloader(
            loc,
            output_dir,
            variable,
        )
        try:
            output_path = downloader.run()
            number_of_successful_locations += 1
        except Exception as ex:
            logger.error(str(ex))
            continue
    logger.info(f'The number of locations which has been correctly downloaded'
                 f' is {number_of_successful_locations} out of'
                 f' {len(locations_df)} for variable {variable}')
    logger.info('Process finished!')
