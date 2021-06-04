from src.visualization.visualize import StationTemporalSeriesPlotter
from pathlib import Path

import click
import logging


PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', ''], case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path("data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
def main_line(varname, country, data_path, metadata_path, station):
    """ Generates a plot for the variable specified for all stations located in 
    the country chosen.

    Args:
        varname (str): variable to plot. Options are: pm25, o3 and no2.
        country (str): country whose stations will be considered.
        data_path (Path): path to the folder containing the data.
        metadata_path (Path): path to the folder containing the metadata.
        station (str): whether to plot any particular station or not.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    StationTemporalSeriesPlotter(
        varname,
        country,
        data_path,
        metadata_path, 
        [station] if station else station
    ).plot_data()


@click.command()
@click.argument('varname', type=click.Choice(['pm25', 'o3', ''], case_sensitive=True))
@click.argument('country', type=click.STRING)
@click.option('-d', '--data_path', type=PATH, required=True)
@click.option('-m', '--metadata_path', type=PATH,
              default=Path("data/external/stations.csv"))
@click.option('-s', '--station', type=click.STRING, default=None)
def main_corrs(varname, country, data_path, metadata_path, station):
    """ Generates a figure showing the correlation between all the features and 
    the forecast bias.

    Args:
        varname (str): variable to plot. Options are: pm25, o3 and no2.
        country (str): country whose stations will be considered.
        data_path (Path): path to the folder containing the data.
        metadata_path (Path): path to the folder containing the metadata.
        station (str): whether to plot any particular station or not.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    StationTemporalSeriesPlotter(
        varname,
        country,
        data_path,
        metadata_path, 
        [station] if station else station
    ).plot_correlations()
