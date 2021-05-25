from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from math import sin, cos, sqrt, atan2, radians

import pandas as pd
import numpy as np
import os
import openaq
import datetime
import logging
import warnings

warnings.filterwarnings("ignore")

tol = 1e-3


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
    distance: float = 0.0

    def __str__(self):
        return f'Location(location_id={self.location_id}, ' \
               f'city={self.city}, country={self.country}, ' \
               f'latitude={self.latitude}, longitude={self.longitude}, ' \
               f'with a distance to the nearest OpenAQ ' \
               f'station of {self.distance} km)'
    

class OpenAQDownloader:
    """
    Class to download data from the OpenAQ platform
    for a specific location of interest
    """
    def __init__(
            self,
            location: Location,
            output_dir: Path,
            variable: str,
            time_range: Dict[str, str] = None
    ):
        self.api = openaq.OpenAQ(version='v2')
        if time_range is None:
            time_range = dict(start='2019-01-01', end='2021-03-31')
        self.location = location
        self.time_range = time_range
        self.output_dir = output_dir
        if variable in ['o3', 'no2', 'so2', 'pm10', 'pm25']:
            self.variable = variable
        else:
            raise NotImplementedError(f"The variable {variable} do"
                                      f" not correspond to any known one")
        self.downloaded_time_range = {}

    def run(self) -> (Path, Path):
        """
        Main method to download data from OpenAQ.
        """
        stations = self.get_closest_stations_to_location()
        data = pd.DataFrame()
        for station in stations.iterrows():
            try:
                data = self.get_data(station[1])
                self.location.distance = station[1]['distance']
                break
            except Exception as ex:
                continue
        output_path_data = self.get_output_path()
        output_path_metadata = self.get_output_path(is_metadata=True)
        logging.info(f'Nearest station is located at'
                     f' {self.location.distance} km')
        self.save_data_and_metadata(data,
                                    output_path_data,
                                    output_path_metadata)
        logging.info(f"Data has been correctly downloaded"
                     f" in {str(output_path_data)}")
        return output_path_data, output_path_metadata

    def get_closest_stations_to_location(self) -> pd.DataFrame:
        """
        Method to check which station is closer to the point of interest.
        """
        stations = self._get_closest_stations_to_location()
        # If not, we calculate the distances to that point
        distances = []
        is_in_temporal_range = []
        for station in stations.iterrows():
            station = station[1]
            distance = get_distance_between_two_points_on_earth(
                station['coordinates.latitude'],
                self.location.latitude,
                station['coordinates.longitude'],
                self.location.longitude
            )
            distances.append(distance)
            if pd.to_datetime(
                    self.time_range['start']
            ) >= station['firstUpdated'].tz_convert(None) and\
                    pd.to_datetime(
                        self.time_range['end']
                    ) <= station['lastUpdated'].tz_convert(None):
                is_in_temporal_range.append(1)
            else:
                is_in_temporal_range.append(0)
        stations['distance'] = distances
        stations['is_in_temporal_range'] = is_in_temporal_range
        stations = stations.sort_values('distance')
        stations = stations.sort_values('is_in_temporal_range', ascending=False)
        return stations

    def _get_closest_stations_to_location(self) -> pd.DataFrame:
        """
        Method to check whether there are stations or not in the city
        where the location of interest is located.
        """
        stations = self.api.locations(
            parameter=self.variable,
            coordinates=f"{self.location.latitude},"
                        f"{self.location.longitude}",
            nearest=4, radius=100000, df=True)

        if len(stations) == 0:
            raise Exception('There are no stations next to'
                            ' this location in OpenAQ for the'
                            ' variable of interest')

        if len(stations[stations['sensorType'] == 'reference grade']) >= 1:
            stations = stations[stations['sensorType'] == 'reference grade']
        return stations

    def get_output_path(self, is_metadata: bool = False):
        """
        Method to get the output paths where the data and metadata are stored.
        """
        city = self.location.city.lower().replace(' ', '_')
        country = self.location.country.lower().replace(' ', '_')
        station_id = self.location.location_id.lower()
        variable = self.variable
        time_range = '_'.join(
            self.downloaded_time_range.values()
        ).replace('-', '')
        ext = '_metadata.csv' if is_metadata else '.csv'
        output_path = Path(
            self.output_dir,
            country,
            city,
            station_id,
            variable,
            f"{variable}_{country}_{city}_{station_id}_{time_range}{ext}"
        )
        return output_path

    def get_data(self, station: pd.Series) -> pd.DataFrame:
        """
        This methods retrieves data from the OpenAQ platform in pd.DataFrame 
        format. The specific self.time_range, given by the user, is selected 
        afterwards. It also takes out every value under 0 (which are considered 
        as NaN).
        """
        self.check_variable_in_station(station)
        try:
            data = self.api.measurements(
                location_id=station['id'],
                parameter=self.variable,
                limit=10000,
                value_from=0,
                date_from=self.time_range['start'],
                date_to=self.time_range['end'],
                index='utc',
                df=True)
        except Exception as ex:
            raise Exception('There is no data in the time range considered for'
                            ' this location of interest')
        data = data.sort_index()
        self.downloaded_time_range['start'] = datetime.datetime.strftime(
            pd.to_datetime(data.index.values[0]),
            '%Y%m%d')
        self.downloaded_time_range['end'] = datetime.datetime.strftime(
            pd.to_datetime(data.index.values[-1]),
            '%Y%m%d')
        return data

    def check_variable_in_station(self, station: pd.Series):
        """
        This method checks whether the stations has available data for the
        variable of interest
        """
        if self.variable not in [x['parameter'] for x in station['parameters']]:
            raise Exception('The variable intended to download is not'
                            ' available for the nearest / exact location')

    def save_data_and_metadata(
            self,
            data: pd.DataFrame,
            output_path_data: Path,
            output_path_metadata: Path
    ):
        """
        This function saves the data (.csv format) or the metadata (.txt format)
        """
        # Directory initialization if they do not exist
        if not output_path_data.parent.exists():
            os.makedirs(output_path_data.parent, exist_ok=True)
        if not output_path_metadata.parent.exists():
            os.makedirs(output_path_metadata.parent, exist_ok=True)
        # Store data in [datetime, value] format
        data['value'].to_csv(output_path_data)
        # Store metadata in [variable, units,
        dict_metadata = {
            'variable': self.variable,
            'units': np.unique(data['unit'].values),
            'latitude_openaq': np.unique(data['coordinates.latitude'].values),
            'longitude_openaq': np.unique(data['coordinates.longitude'].values),
            'total_values': len(data),
            'initial_time': data.index.values[0],
            'final_time': data.index.values[-1],
            'distance': self.location.distance
        }
        metadata = pd.DataFrame(dict_metadata)
        metadata.to_csv(output_path_metadata)


def get_distance_between_two_points_on_earth(
        lat1: float,
        lat2: float,
        lon1: float,
        lon2: float
):
    """
    Function to calculate the distance between an station from OpenAQ
    and the coordinates of interest. Use Haversine distance.
    """
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

