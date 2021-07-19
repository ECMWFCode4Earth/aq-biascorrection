import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, TypeVar

import xarray as xr
import pandas as pd
import requests
from tenacity import retry

from src.constants import ROOT_DIR


from src.logging import get_logger

logger = get_logger("Data utilities")

L = TypeVar("L", bound="Location")


@dataclass
class Location:
    """
    Class to define specific location of interest
    with its correspondent attributes
    """

    location_id: str
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    elevation: float

    def __str__(self):
        return (
            f"Location(location_id={self.location_id}, "
            f"city={self.city}, country={self.country}, "
            f"latitude={self.latitude}, longitude={self.longitude}, "
            f"elevation={self.elevation}, timezone={self.timezone}"
        )

    def get_observations_path(self, directory, variable, time_range):
        """
        Method to get the observations path given:
        Location, directory, variable, time_range.
        """
        city = self.city.lower().replace(" ", "-")
        country = self.country.lower().replace(" ", "-")
        station_id = self.location_id.lower()
        ext = ".nc"
        output_path = Path(
            directory,
            country,
            city,
            station_id,
            variable,
            f"{variable}_{country}_{city}_{station_id}_{time_range}{ext}",
        )
        return output_path

    def get_forecast_path(self, directory, time_range):
        """
        Method to get the forecast path given:
        Location, directory, variable, time_range.
        """
        city = self.city.lower().replace(" ", "-")
        country = self.country.lower().replace(" ", "-")
        station_id = self.location_id.lower()
        ext = ".nc"
        output_path = Path(
            directory,
            country,
            city,
            station_id,
            f"cams_{country}_{city}_{station_id}_{time_range}{ext}",
        )
        return output_path

    @classmethod
    def get_location_by_id(cls, location_id: str) -> L:
        """Get the Location object from the station id.

        Args:
            location_id (str): ID of the location. Style is AA000: two capital letters,
            followed by three numbers.

        Returns:
            Location: Location object corresponding to the ID.
        """
        metadata = pd.read_csv(
            ROOT_DIR / "data/external/stations.csv",
            index_col=0,
            usecols=list(range(1, 8)),
        )
        vals = metadata.loc[location_id].values
        loc = cls(location_id, *vals)
        return loc


@retry
def get_elevation_for_location(latitude: float, longitude: float):
    """
    Function to get the elevation of a specific location given the latitude and
    the longitude
    """
    url = (
        f"https://api.open-elevation.com/api/v1/lookup?"
        f"locations={round(latitude, 4)},{round(longitude, 4)}"
    )
    elevation = requests.get(url, timeout=30).json()["results"][0]["elevation"]
    return elevation


def get_countries(data_path: Path = ROOT_DIR / "data/external") -> List[str]:
    """Get all the countries with stations available."""
    df = pd.read_csv(data_path / "stations.csv", usecols=["country"])
    return list(df.country.values)


# Utilities for I/O disck operations
def write_netcdf(output_path: Path, ds: xr.Dataset):
    if not output_path.parent.exists():
        os.makedirs(output_path.parent, exist_ok=True)
    comp = dict(zlib=True, complevel=1, shuffle=True)
    encoding = {var: comp for var in ds.data_vars}
    statistics = []
    for var in list(ds.data_vars):
        mean = float(ds[var].mean())
        std = float(ds[var].std())
        statistics.append(f"{var}= {mean:.4f} ({std:.4f} std)")
    logger.info("\n".join(statistics))

    ds.to_netcdf(path=output_path, unlimited_dims=None, encoding=encoding)


def remove_intermediary_paths(intermediary_paths: List[Path]):
    for path in intermediary_paths:
        if path.exists():
            os.remove(path)
