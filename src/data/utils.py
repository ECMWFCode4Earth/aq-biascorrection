from dataclasses import dataclass
from pathlib import Path
from tenacity import retry
from typing import List
from src.constants import ROOT_DIR
import pandas as pd
import requests


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
        return f'Location(location_id={self.location_id}, ' \
               f'city={self.city}, country={self.country}, ' \
               f'latitude={self.latitude}, longitude={self.longitude}, ' \
               f'elevation={self.elevation}, timezone={self.timezone}'

    def get_observations_path(self, directory, variable, time_range):
        """
        Method to get the observations path given:
        Location, directory, variable, time_range.
        """
        city = self.city.lower().replace(' ', '-')
        country = self.country.lower().replace(' ', '-')
        station_id = self.location_id.lower()
        ext = '.nc'
        output_path = Path(
            directory,
            country,
            city,
            station_id,
            variable,
            f"{variable}_{country}_{city}_{station_id}_{time_range}{ext}"
        )
        return output_path

    def get_forecast_path(self, directory, time_range):
        """
        Method to get the forecast path given:
        Location, directory, variable, time_range.
        """
        city = self.city.lower().replace(' ', '-')
        country = self.country.lower().replace(' ', '-')
        station_id = self.location_id.lower()
        ext = '.nc'
        output_path = Path(
            directory,
            country,
            city,
            station_id,
            f"cams_{country}_{city}_{station_id}_{time_range}{ext}"
        )
        return output_path

@retry
def get_elevation_for_location(latitude: float, longitude:float):
    """
    Function to get the elevation of a specific location given the latitude and
    the longitude
    """
    url = f"https://api.open-elevation.com/api/v1/lookup?" \
          f"locations={round(latitude, 4)},{round(longitude, 4)}"
    elevation = requests.get(url, timeout=30).json()['results'][0]['elevation']
    return elevation


def get_countries(data_path: Path = ROOT_DIR / "data/external") -> List[str]:
    """Get all the countries with stations available. """
    df = pd.read_csv(data_path / "stations.csv", usecols=['country'])
    return list(df.country.values)