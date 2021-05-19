from pathlib import Path
from downloader import Location, OpenAQDownloader

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
    for the respective varaible / location combination and  stores the
    data at the output directory (given as an argument)
    """
    locations_df = pd.read_csv(csv_path)
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
            output_path, output_path_metadata = downloader.run()
        except Exception as ex:
            logging.error(str(ex))
            continue


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    download_openaq_data_from_csv_with_locations_info(
        Path('../../../data/external/stations.csv'),
        Path('../../../data/raw/observations/'),
        'o3'
    )

