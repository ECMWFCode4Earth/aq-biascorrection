import logging
import os
import sys
from pathlib import Path

import click
from pytz import NonExistentTimeError

from src import constants
from src.data import utils
from src.visualization.data_visualization import StationTemporalSeriesPlotter

PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument(
    "varname", type=click.Choice(["pm25", "o3", "no2", "all"], case_sensitive=True)
)
@click.argument("country", type=click.STRING)
@click.option("-d", "--data_path", type=PATH, required=True)
@click.option(
    "-m",
    "--metadata_path",
    type=PATH,
    default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"),
)
@click.option("-s", "--station", type=click.STRING, default=None)
@click.option(
    "-o",
    "--output_path",
    type=PATH,
    default=None,
    help="Output path of the figure to be saved.",
)
@click.option(
    "-a",
    "--agg_by",
    type=click.Choice(["daily", "monthly"]),
    default=None,
    help="Indicates if any temporal aggregation is" " needed.",
    metavar="<str>",
)
def main_line(
    varname: str,
    country: str,
    data_path: Path,
    metadata_path: Path,
    station: str,
    output_path: Path = None,
    agg_by: str = None,
):
    """Generates a plot for the variable specified for all stations located in
    the country chosen.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'. \n
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    countries = utils.get_countries() if country == "all" else [country]
    varnames = ["pm25", "o3", "no2"] if varname == "all" else [varname]
    for var in varnames:
        for country in countries:
            temp_agg = agg_by.capitalize() if agg_by else "Raw"
            if output_path is not None:
                output_folder = output_path / var / "StationBias" / temp_agg
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = None
            logging.info(f"Processing plot for {var} bias for {country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path,
                [station] if station else station,
            ).plot_data(output_folder, agg=agg_by)
    logging.info("The script finished!")


@click.command()
@click.argument(
    "varname", type=click.Choice(["pm25", "o3", "no2", "all"], case_sensitive=True)
)
@click.argument("country", type=click.STRING)
@click.option("-d", "--data_path", type=PATH, required=True)
@click.option(
    "-m",
    "--metadata_path",
    type=PATH,
    default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"),
)
@click.option("-s", "--station", type=click.STRING, default=None)
@click.option(
    "-o",
    "--output_path",
    type=PATH,
    default=None,
    help="Output path of the figure to be saved.",
)
@click.option(
    "-a",
    "--agg_by",
    type=click.Choice(["daily", "monthly"]),
    default=None,
    help="Indicates if any temporal aggregation is" " needed.",
    metavar="<str>",
)
def main_corrs(
    varname: str,
    country: str,
    data_path: Path,
    metadata_path: Path,
    station: str,
    output_path: Path = None,
    agg_by: str = None,
):
    """Generates a figure showing the correlation between all the features and
    the forecast bias.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'. \n
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    countries = utils.get_countries() if country == "all" else [country]
    varnames = ["pm25", "o3", "no2"] if varname == "all" else [varname]
    for var in varnames:
        for country in countries:
            temp_agg = agg_by.capitalize() if agg_by else "Raw"
            if output_path is not None:
                output_folder = output_path / var / "Correlations" / temp_agg
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = None
            logging.info(f"Processing correlation with {var} bias for " f"{country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path,
                [station] if station else station,
            ).plot_correlations(output_folder, agg=agg_by)
        logging.info("The script finished!")


@click.command()
@click.argument(
    "varname", type=click.Choice(["pm25", "o3", "no2", "all"], case_sensitive=True)
)
@click.argument("country", type=click.STRING)
@click.option("-d", "--data_path", type=PATH, required=True)
@click.option(
    "-m",
    "--metadata_path",
    type=PATH,
    default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"),
)
@click.option(
    "--show_std",
    type=click.BOOL,
    default=True,
    help="Show the " "estimated standard deviation of the dataset.",
)
@click.option(
    "-o",
    "--output_path",
    type=PATH,
    default=None,
    help="Output path of the figure to be saved.",
)
def main_hourly_bias(
    varname: str,
    country: str,
    data_path: Path,
    metadata_path: Path,
    show_std: bool = True,
    output_path: Path = None,
):
    """Generates a figure showing the correlation between all the features and
    the forecast bias.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'.\n
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    countries = utils.get_countries() if country == "all" else [country]
    varnames = ["pm25", "o3", "no2"] if varname == "all" else [varname]
    for var in varnames:
        for country in countries:
            logging.info(f"Processing hourly {var} bias for {country}.")
            if output_path is not None:
                output_folder = output_path / var / "HourlyBias"
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = None
            StationTemporalSeriesPlotter(
                var, country, data_path, metadata_path
            ).plot_hourly_bias(show_std, output_folder)
    logging.info("The script finished!")


@click.command()
@click.argument(
    "varname", type=click.Choice(["pm25", "o3", "no2", "all"], case_sensitive=True)
)
@click.argument("country", type=click.STRING)
@click.option("-d", "--data_path", type=PATH, required=True)
@click.option(
    "-m",
    "--metadata_path",
    type=PATH,
    metavar="<str>",
    default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"),
)
@click.option(
    "-o",
    "--output_path",
    type=PATH,
    default=None,
    metavar="<str>",
    help="Output path of the figure to be saved.",
)
@click.option(
    "-a",
    "--agg_by",
    type=click.Choice(["daily", "monthly"]),
    default=None,
    help="Indicates if any temporal aggregation is" " needed.",
    metavar="<str>",
)
def main_cdf_bias(
    varname: str,
    country: str,
    data_path: Path,
    metadata_path: Path,
    output_path: Path = None,
    agg_by: str = None,
):
    """Generates a figure showing the empirical Cumulative Distribution
    Function (CDF) of the bias observed at each station.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'.\n
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    countries = utils.get_countries() if country == "all" else [country]
    varnames = ["pm25", "o3", "no2"] if varname == "all" else [varname]
    for var in varnames:
        for country in countries:
            temp_agg = agg_by.capitalize() if agg_by else "Raw"
            if output_path is not None:
                output_folder = output_path / var / "BiasDistribution" / temp_agg
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = None
            logging.info(f"Processing CDF of {var} bias for {country}.")
            StationTemporalSeriesPlotter(
                var, country, data_path, metadata_path
            ).plot_bias_cdf(output_folder, agg=agg_by)
    logging.info("The script finished!")


@click.command()
@click.argument(
    "varname", type=click.Choice(["pm25", "o3", "no2", "all"], case_sensitive=True)
)
@click.argument("country", type=click.STRING)
@click.option("-d", "--data_path", type=PATH, required=True)
@click.option(
    "-m",
    "--metadata_path",
    type=PATH,
    default=Path(f"{constants.ROOT_DIR}/data/external/stations.csv"),
)
@click.option("-s", "--station", type=click.STRING, default=None)
@click.option(
    "-o",
    "--output_path",
    type=PATH,
    default=None,
    help="Output path of the figure to be saved.",
)
def main_monthly_bias(
    varname: str,
    country: str,
    data_path: Path,
    metadata_path: Path,
    station: str,
    output_path: Path = None,
):
    """Generates a plot for the variable specified for all stations located in
    the country chosen. This plots shows the montly distribution of the bias.

    Args:

        varname (str): The name of the variable to consider. Specify 'all' for
        selecting all variables. Choiches are: 'pm25', 'o3', 'no2'. \n
        country (str): The country to consider. Specify 'all' for selecting
        all countries with processed data.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)
    countries = utils.get_countries() if country == "all" else [country]
    varnames = ["pm25", "o3", "no2"] if varname == "all" else [varname]
    for var in varnames:
        for country in countries:
            if output_path is not None:
                output_folder = output_path / var / "MonthlyBias"
                os.makedirs(output_folder, exist_ok=True)
            else:
                output_folder = None
            logging.info(f"Processing bias plot for {var} bias for {country}.")
            StationTemporalSeriesPlotter(
                var,
                country,
                data_path,
                metadata_path,
                [station] if station else station,
            ).plot_monthly_bias(output_folder)
    logging.info("The script finished!")
