import os
import openaq
from pathlib import Path
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians


class Location:
    """
    Class to define specific location of interest
    with its correspondent attributes
    """
    def __init__(
            self,
            location_id: str,
            city: str,
            country: str,
            latitude: float,
            longitude: float,
    ):
        self.location_id = location_id
        self.latitude = round(latitude, 2)
        self.longitude = round(longitude, 2)
        self.city = city
        self.country = country


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
            time_range=None
    ):
        self.api = openaq.OpenAQ()
        if time_range is None:
            time_range = dict(start='2019-01-01', end='2021-03-31')
        self.loc = location
        self.time_range = time_range
        self.output_dir = output_dir
        if variable in ['o3', 'no2', 'so2', 'pm10', 'pm25']:
            self.variable = variable
        else:
            raise NotImplementedError(f"The variable {variable} do"
                                      f" not correspond to any known one")

    def run(
            self
    ) -> str:
        """
        Main method to download data from OpenAQ.
        """
        output_path_data = self.get_output_path()
        output_path_metadata = self.get_output_path(is_metadata=True)
        station = self.get_closest_station_to_location()
        data = self.get_data(station)
        self.save_data_and_metadata(data, output_path_data, output_path_metadata)
        return f"Data has been correctly downloaded in {str(output_path_data)}"

    def get_closest_station_to_location(
            self
    ) -> pd.Series:
        """
        Method to check which station is closer to the point of interest.
        """
        stations = self.get_stations_in_city()
        # First of all, we check if any of the stations match the
        # exact location of the point of interest
        if len(stations[stations['is_in_location']==True]):
            station = stations[stations['is_in_location']==True].iloc[0]
        # If not, we calculate the distances to that point
        else:
            distances = []
            for station in stations.iterrows():
                station = station[1]
                station_lat = round(station['coordinates.latitude'], 2)
                station_lon = round(station['coordinates.longitude'], 2)
                distance = get_distance_between_two_points_on_earth(
                    station_lat,
                    self.loc.latitude,
                    station_lon,
                    self.loc.longitude
                )
                distances.append(distance)
            stations['distance'] = distances
            station = stations.iloc[stations['distance'].idxmax()]
        self.check_variable_in_station(station)
        return station

    def get_stations_in_city(
            self
    ) -> pd.DataFrame:
        """
        Method to check whether there are stations or not in the city
        where the location of interest is located.
        """
        stations = self.api.locations(
            city=self.loc.city,
            df=True
        )
        is_in_location = []
        for station in stations.iterrows():
            station = station[1]
            station_lat = round(station['coordinates.latitude'], 2)
            station_lon = round(station['coordinates.longitude'], 2)
            if station_lat != self.loc.latitude or station_lon != self.loc.longitude:
                print('The OpenAQ station coordinates do not match'
                      ' with the location of interest coordinates')
                is_in_location.append(False)
            else:
                is_in_location.append(True)
        stations['is_in_location'] = is_in_location
        return stations

    def get_output_path(self, is_metadata=False):
        """
        Method to get the output paths where the data and metadata are stored.
        """
        city = self.loc.city.lower()
        station_id = self.loc.location_id.lower()
        variable = self.variable
        time_range = f"{self.time_range['start'].replace('-', '')}_{self.time_range['end'].replace('-', '')}"
        if not is_metadata:
            output_path = Path(
                self.output_dir,
                city,
                station_id,
                f"{variable}_{city}_{station_id}_{time_range}.csv"
            )
        else:
            output_path = Path(
                self.output_dir,
                city,
                station_id,
                f"{variable}_{city}_{station_id}_{time_range}_metadata.txt"
            )
        return output_path

    def get_data(
            self,
            station
    ) -> pd.DataFrame:
        """
        This methods retrieves data from the OpenAQ platform in pd.DataFrame format.
        The specific self.time_range, given by the user, is selected afterwards. It
        also takes out every value under 0 (which are considered as NaN).
        """
        data = self.api.measurements(
            city=station['city'],
            location=station['location'],
            parameter=self.variable,
            limit=100000,
            df=True)
        data_in_time = data[
            (data['date.utc'] > self.time_range['start']) & (data['date.utc'] <= self.time_range['end'])
        ]
        # The date.utc columns is set as index in order to have always the
        # same time reference
        data_in_time.reset_index(inplace=True)
        data_in_time.set_index('date.utc', inplace=True)
        data_in_time = data_in_time[data_in_time['value'] >= 0]
        return data_in_time

    def check_variable_in_station(
            self,
            station: pd.Series
    ):
        """
        This method checks whether the stations has available data for the
        variable of interest
        """
        if self.variable not in station['parameters']:
            raise Exception('The variable intended to download is not'
                            ' available for the nearest / exact location')

    def save_data_and_metadata(
            self,
            data: pd.DataFrame,
            output_path_data: Path,
            output_path_metadata: Path
    ):
        """
        This function saves the data in .csv format and the metadata in .txt format
        """
        if not output_path_data.parent.exists():
            os.makedirs(output_path_data.parent, exist_ok=True)
        if not output_path_metadata.parent.exists():
            os.makedirs(output_path_metadata.parent, exist_ok=True)
        data['value'].to_csv(output_path_data)
        metadata = f"For the parameter {self.variable} in units of {np.unique(data['unit'].values)}, " \
                   f"there is a total of {len(data)} values ranging from {data.index.values[0]} " \
                   f"to {data.index.values[-1]}. All of them with a value equal or above 0."
        with open(output_path_metadata, 'w') as f:
            f.write(metadata)
            f.close()


def get_distance_between_two_points_on_earth(
        lat1,
        lat2,
        lon1,
        lon2
):
    """
    Function to calculate the distance between an station from OpenAQ
    and the coordinates of interest
    """
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


if __name__ == '__main__':
    loc = Location("AE001", "Dubai", "United Arab Emirates", 25.0657, 55.17128)
    OpenAQDownloader(loc, Path('/predictia-nas2/Data/'), 'o3').run()