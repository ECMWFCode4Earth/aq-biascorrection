# -*- coding: utf-8 -*-
from src.scripts.extraction_openaq import download_openaq_data_from_csv_with_locations_info
from src.data.extraction.cams_forecast import CAMSProcessor
from src.data.transformation.transformation_data import DataTransformer
from src import constants
from pathlib import Path

import click
import sys
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
def main(
    variable: str,
    locations_csv_path: Path,
    output_observation_extraction: Path,
    input_forecast_extraction: Path,
    intermediary_forecast_extraction: Path,
    output_forecast_extraction: Path,
    output_data_transformation: Path
):
    """
    Function to do the whole ETL process
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    logger = logging.getLogger("ETL Pipeline")
    logger.info('Making final data set from raw data')

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
    logger.info('Process finished!')
