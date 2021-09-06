from pathlib import Path

import click
import pandas as pd

from src.constants import ROOT_DIR
from src.models.validation import Validator

PATH = click.Path(exists=True, path_type=Path)
DATE_TYPE = click.DateTime()

from src.logger import get_logger

logger = get_logger("Model trainer")


@click.command()
@click.argument("variable", type=click.Choice(["pm25", "no2", "o3", "all"]))
@click.argument("model_name", type=click.STRING)
@click.option(
    "-l",
    "--locations_csv_path",
    type=PATH,
    help="Path to " "the file where the locations of interest are defined",
    required=True,
)
@click.option(
    "-ov",
    "--output_dir_visualizations",
    type=PATH,
    help="Path to the directory where the visualizations are stored as images",
    required=True,
    default=ROOT_DIR / "reports" / "figures" / "results",
)
@click.option(
    "-om",
    "--output_dir_metrics",
    type=PATH,
    help="Path to the directory where the metrics are stored as tables",
    required=True,
    default=ROOT_DIR / "reports" / "tables" / "results",
)
def main(
    variable: str,
    model_name: str,
    locations_csv_path: Path,
    output_dir_visualizations: Path,
    output_dir_metrics: Path,
):
    """
    Script to validate the model results for every location of interest.
    """

    stations = pd.read_csv(locations_csv_path)
    for station_id in stations["id"].values:
        try:
            Validator(
                model_name, variable, output_dir_visualizations, output_dir_metrics
            ).run(station_id)
        except:
            pass
