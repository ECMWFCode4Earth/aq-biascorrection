import logging
import sys
from pathlib import Path

import click

from src import constants
from src.constants import ROOT_DIR
from src.data.transformer import DataTransformer

PATH = click.Path(exists=True, path_type=Path)


@click.command()
@click.argument("var", type=click.Choice(["pm25", "no2", "o3", "all"]))
@click.option(
    "-l",
    "--locations_csv_path",
    type=PATH,
    default=ROOT_DIR / "data/external/stations.csv",
    help="Path to the file where the locations " "of interest are defined",
)
@click.option(
    "-o",
    "--output_dir",
    type=PATH,
    default=ROOT_DIR / "data/processed/",
    help="Output directory where to store the data to",
)
def main(
    var: str,
    locations_csv_path: Path = ROOT_DIR / "data/external/stations.csv",
    output_dir: Path = ROOT_DIR / "data/processed/",
):
    """Transform data.

    VAR is the variable to transform.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=constants.log_fmt)

    if var == "all":
        variables = ["no2", "o3", "pm25"]
    else:
        variables = [var]

    for variable in variables:
        logging.info(f"Transforming {variable} variable for all locations.")
        DataTransformer(variable, locations_csv_path, output_dir).run()

    logging.info("Process finished!")
