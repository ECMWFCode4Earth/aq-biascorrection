from dataclasses import dataclass
from pathlib import Path


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
               f'latitude={self.latitude}, longitude={self.longitude}'

    def get_observations_path(self, directory, variable, time_range):
        """
        Method to get the output path where the data is stored.
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
        Method to get the output path where the data is stored.
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