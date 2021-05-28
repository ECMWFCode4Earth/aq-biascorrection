import logging
import argparse
from pathlib import Path

from src.data.preprocess.preprocessor import CAMSProcessor


parser = argparse.ArgumentParser(
    prog='CAMSProcessor',
)

parser.add_argument(
    "-input", '--input_dir',
    help="Input directory where to take the data from"
)

parser.add_argument(
    "-output", '--output_dir',
    help="Output directory where to store the data to"
)

parser.add_argument(
    "-locations", '--locations_csv_path',
    help="Path to the file where the locations of interest are defined"
)

parser.add_argument(
    "-period", '--time_period',
    default=None,
    help="Period of time in which to process the CAMS data"
)

args = parser.parse_args()
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

CAMSProcessor(
    Path(args.input_dir),
    Path(args.locations_csv_path),
    Path(args.output_dir),
    args.time_period
).run()
