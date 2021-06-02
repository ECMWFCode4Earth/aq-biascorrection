from pathlib import Path
from src.data.extraction.openaq_obs import OpenAQDownloader
from src.data.utils import Location

import pandas as pd
import logging
import argparse


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
        logging.info(f"Starting process for location of"
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
            logging.error(str(ex))
            continue
    logging.info(f'The number of locations which has been correctly downloaded'
                 f' is {number_of_successful_locations} out of'
                 f' {len(locations_df)} for variable {variable}')
    logging.info('Process finished!')


parser = argparse.ArgumentParser(
    prog='OpenAQDownloader',
)

parser.add_argument(
    "-output", '--output_dir',
    help="Output directory where to store the data to"
)

parser.add_argument(
    "-locations", '--locations_csv_path',
    help="Path to the file where the locations of interest are defined"
)

parser.add_argument(
    "-var", '--variable',
    default=None,
    help="Variable to which to extraction the OpenAQ data"
)

args = parser.parse_args()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

download_openaq_data_from_csv_with_locations_info(
    Path(args.locations_csv_path),
    Path(args.output_dir),
    args.variable
)

