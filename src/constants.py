import os
from pathlib import Path

str2agg = {"daily": "D", "monthly": "M"}


var2longstr = {
    "no2": "Nitrogen dioxide",
    "o3": "Ozone",
    "pm25": "Particulate matter (PM2.5)",
}


units2str = {"µg/m³": "microgram / m^3", "ppm": "ppm"}


def project_root(dir=None):
    newdir = dir.parent if dir is not None else Path(__file__).parent
    if len(list(newdir.glob("README.md"))):
        return newdir
    else:
        return project_root(newdir)


ROOT_DIR = project_root()

print(f"Project directory (ROOT_DIR): '{ROOT_DIR}'")

log_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
