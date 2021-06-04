# -*- coding: utf-8 -*-
from src.scripts.extraction_openaq import download_openaq_data_by_locations_csv
from src.data.extraction.cams_forecast import CAMSProcessor
from src.data.transformation.transformation_data import DataTransformer
from pathlib import Path

import click
import logging


@click.command()
@click.argument('variable', type=click.STRING)
@click.argument('locations_csv_path', type=click.Path(path_type=Path))
@click.argument('output_observation_extraction', 
                type=click.Path(exists=True, path_type=Path))
@click.argument('input_forecast_extraction', 
                type=click.Path(exists=True, path_type=Path))
@click.argument('intermediary_forecast_extraction', 
                type=click.Path(exists=True, path_type=Path))
@click.argument('output_forecast_extraction', 
                type=click.Path())
@click.argument('output_data_transformation', 
                type=click.Path())
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
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    download_openaq_data_by_locations_csv(
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
    logger.info('Process finished!')
