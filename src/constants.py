import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

str2agg = {
    'daily': 'D',
    'monthly': 'M'
}


var2longstr = {
    'no2': 'Nitrogen dioxide',
    'o3': 'Ozone',
    'pm25': 'Particulate matter (PM2.5)'
}


units2str = {
    "µg/m³": "microgram / m^3",
    "ppm": "ppm"
}


ROOT_DIR = Path(os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath("setup.py"))))

log_fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'