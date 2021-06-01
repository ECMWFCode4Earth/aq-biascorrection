from dataclasses import dataclass
from pathlib import Path

import googlemaps
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

    def __str__(self):
        return f'Location(location_id={self.location_id}, ' \
               f'city={self.city}, country={self.country}, ' \
               f'latitude={self.latitude}, longitude={self.longitude}, ' \
               f'timezone={self.timezone}'

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

    def get_height_for_location(self, api_key: str):
        url = f"https://api.open-elevation.com/api/v1/lookup?" \
              f"locations={round(self.latitude, 2)},{round(self.longitude, 2)}"
        elevation = requests.get(url).json()['results'][0]['elevation']
        return elevation
