from src.visualization.visualize import StationTemporalSeriesPlotter
from src.data import utils
from src import constants
from pathlib import Path

import click
import logging
import sys


PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2', 'all'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
@click.option('-o', '--output_path', type=PATH, 
              default=None, help="Output path of the figure to be saved.")
def main_line(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    station: str,
    output_path: Path = None
):
    """ Generates a plot for the variable specified for all stations located in 
    the country chosen.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for 
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'.
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    countries = utils.get_countries() if country == 'all' else [country]
    varnames = ['pm25', 'o3', 'no2'] if varname == 'all' else [varname]
    for var in varnames:
        for country in countries:
            logging.info(f"Processing plot for {varname} bias for {country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path, 
                [station] if station else station
            ).plot_data(output_path)
    logging.info("The script finished!")


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2', 'all'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
@click.option('-o', '--output_path', type=click.Path(writable=True), 
              default=None, help="Output path of the figure to be saved.")
def main_corrs(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    station: str,
    output_path: Path = None
):
    """ Generates a figure showing the correlation between all the features and 
    the forecast bias.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for 
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'.
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    
    countries = utils.get_countries() if country == 'all' else [country]
    varnames = ['pm25', 'o3', 'no2'] if varname == 'all' else [varname]
    for var in varnames:
        for country in countries:
            logging.info(f"Processing correlation with {varname} bias for "
                        f"{country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path, 
                [station] if station else station
            ).plot_correlations(output_path)
        logging.info("The script finished!")


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2', 'all'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('--show_std', type=click.BOOL, default=True, help="Show the "
              "estimated standard deviation of the dataset.")
@click.option('-o', '--output_path', type=PATH, default=None, 
              help="Output path of the figure to be saved.")
def main_hourly_bias(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    show_std: bool = True,
    output_path: Path = None,   
):
    """ Generates a figure showing the correlation between all the features and 
    the forecast bias.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for 
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'.
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    
    countries = utils.get_countries() if country == 'all' else [country]
    varnames = ['pm25', 'o3', 'no2'] if varname == 'all' else [varname]
    for var in varnames:
        for country in countries:
            logging.info(f"Processing hourly {varname} bias for {country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path
            ).plot_hourly_bias(show_std, output_path)
    logging.info("The script finished!")
