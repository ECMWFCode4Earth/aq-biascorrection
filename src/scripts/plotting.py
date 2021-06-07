from src.visualization.visualize import StationTemporalSeriesPlotter
from src import constants
from pathlib import Path

import click
import logging
import sys


PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
def main_line(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    station: str
):
    """ Generates a plot for the variable specified for all stations located in 
    the country chosen.

    Options:
        data_path (Path): path to the folder containing the data.
        metadata_path (Path): path to the folder containing the metadata.
        station (str): whether to plot any particular station or not.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    StationTemporalSeriesPlotter(
        varname,
        country,
        data_path,
        metadata_path, 
        [station] if station else station
    ).plot_data()
    logging.info("The script finished!")


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
def main_corrs(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    station: str
):
    """ Generates a figure showing the correlation between all the features and 
    the forecast bias.

    Options:
        data_path (Path): path to the folder containing the data.
        metadata_path (Path): path to the folder containing the metadata.
        station (str): whether to plot any particular station or not.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    StationTemporalSeriesPlotter(
        varname,
        country,
        data_path,
        metadata_path, 
        [station] if station else station
    ).plot_correlations()
    logging.info("The script finished!")


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', 'no2'], 
                                             case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"))
@click.option('--show_std', type=click.BOOL, default=True, help="Show the "
              "estimated standard deviation of the dataset.")
def main_hourly_bias(
    varname: str, 
    country: str, 
    data_path: Path, 
    metadata_path: Path,
    show_std: bool = True
):
    """ Generates a figure showing the correlation between all the features and 
    the forecast bias.

    Options:
        data_path (Path): path to the folder containing the data.
        metadata_path (Path): path to the folder containing the metadata.
        station (str): whether to plot any particular station or not.
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    StationTemporalSeriesPlotter(
        varname,
        country,
        data_path,
        metadata_path
    ).plot_hourly_bias(show_std)
    logging.info("The script finished!")
