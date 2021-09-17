import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

str2agg = {"daily": "D", "monthly": "M"}


var2longstr = {
    "no2": "Nitrogen dioxide",
    "o3": "Ozone",
    "pm25": "Particulate matter (PM2.5)",
}


units2str = {"µg/m³": "microgram / m^3", "ppm": "ppm"}

load_dotenv(find_dotenv())

ROOT_DIR = Path(os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath("setup.py"))))

ADS_API_KEY = os.getenv('ADS_API_KEY')

print(f"Project directory (ROOT_DIR): '{ROOT_DIR}'")

print(f"Your credentials for the ADS (ADS_API_KEY): '{ADS_API_KEY}'")

log_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
