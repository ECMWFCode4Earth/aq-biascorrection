from src.data.transformation.transformation_data import DataTransformer
from pathlib import Path

import argparse
import logging

parser = argparse.ArgumentParser(
    prog='DataTransformer',
)

parser.add_argument(
    "-var", '--variable',
    default=None,
    help="Variable to which to extraction the OpenAQ data"
)

parser.add_argument(
    "-locations", '--locations_csv_path',
    help="Path to the file where the locations of interest are defined"
)

parser.add_argument(
    "-output", '--output_dir',
    help="Output directory where to store the data to"
)


args = parser.parse_args()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

DataTransformer(
    args.variable,
    Path(args.locations_csv_path),
    Path(args.output_dir)
).run()