from pathlib import Path

from src.data.download.downloader import OpenAQDownloader
from src.data.utils import Location

import pandas as pd
import logging


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
            location[1]['longitude']
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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    for variable in ['o3', 'no2', 'pm25']:
        download_openaq_data_from_csv_with_locations_info(
            Path('../../../data/external/stations.csv'),
            Path('../../../data/processed/observations/'),
            variable
        )

