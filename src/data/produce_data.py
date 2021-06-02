# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.scripts.extraction_openaq import download_openaq_data_from_csv_with_locations_info
from src.data.extraction.cams_forecast import CAMSProcessor
from src.data.transformation.transformation_data import DataTransformer


@click.command()
@click.argument('variable', type=click.STRING)
@click.argument('locations_csv_path', type=click.Path())
@click.argument('output_observation_extraction', type=click.Path(exists=True))
@click.argument('input_forecast_extraction', type=click.Path(exists=True))
@click.argument('intermediary_forecast_extraction', type=click.Path(exists=True))
@click.argument('output_forecast_extraction', type=click.Path())
@click.argument('output_data_transformation', type=click.Path())
def main(variable,
         locations_csv_path,
         output_observation_extraction,
         input_forecast_extraction,
         intermediary_forecast_extraction,
         output_forecast_extraction,
         output_data_transformation
         ):
    """
    Function to do the whole ETL process
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    download_openaq_data_from_csv_with_locations_info(
        locations_csv_path,
        output_observation_extraction,
        variable
    )

    CAMSProcessor(
        input_forecast_extraction,
        intermediary_forecast_extraction,
        locations_csv_path,
        output_forecast_extraction,
        None
    ).run()

    DataTransformer(
        variable,
        locations_csv_path,
        output_data_transformation,
        observations_dir=output_observation_extraction,
        forecast_dir=output_forecast_extraction,
        time_range=None
    ).run()




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
