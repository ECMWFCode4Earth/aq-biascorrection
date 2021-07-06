import datetime
import logging
import os
import warnings
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List

import openaq
import pandas as pd
import xarray as xr

from src.data.utils import Location
from src.data import utils
from src.constants import units2str, var2longstr


logger = logging.getLogger("OpenAQ Downloader")
warnings.filterwarnings("ignore")



class OpenAQDownloader:
    """
    Class to extraction data from the OpenAQ platform for a specific location
    of interest.
    It downloads the nearest station with the highest number of measurements
    in the time_range given by the user.
    """
    def __init__(
            self,
            location: Location,
            output_dir: Path,
            variable: str,
            time_range: Dict[str, str] = None
    ):
        self.units = None
        # To take the last version of the OpenAQ API
        self.api = openaq.OpenAQ(version='v2')
        if time_range is None:
            time_range = dict(start='2019-06-01', end='2021-03-31')
        self.location = location
        self.time_range = time_range
        self.output_dir = output_dir
        if variable in ['o3', 'no2', 'so2', 'pm10', 'pm25']:
            self.variable = variable
        else:
            raise NotImplementedError(f"The variable {variable} do"
                                      f" not correspond to any known one")
        self.downloaded_time_range = {}

    def run(self) -> Path:
        """
        Main method for the OpenAQDownloader class.
        """
        output_path_data = self.location.get_observations_path(
            self.output_dir, self.variable,
            '_'.join(self.time_range.values()).replace('-', '')
        )
        if not output_path_data.exists():
            stations = self.get_closest_stations_to_location()
            datasets = self.get_data(stations)
            self.concat_filter_and_save_data(datasets, output_path_data)
            logger.info(f"Data has been correctly downloaded"
                        f" in {str(output_path_data)}")
        else:
            logger.info(f"Data already exists in {str(output_path_data)}")
        return output_path_data

    def get_closest_stations_to_location(self) -> pd.DataFrame:
        """
        Method to return the stations within 100km of the location of interest.
        It returns a pd.DataFrame with two new columns: distance, which
        corresponds with the distance of the location of interest to the OpenAQ
        station, and is_in_temporal_range, which determines whether the
        measurements made by the OpenAQ station are entirely in the time_range
        given by the user.
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
            elif pd.to_datetime(
                    self.time_range['end']
            ) < station['firstUpdated'].tz_convert(None):
                is_in_temporal_range.append(-1)
            else:
                openaq_dates = pd.date_range(
                    datetime.datetime.strftime(
                        station['firstUpdated'].tz_convert(None), '%Y-%m-%d'
                    ),
                    datetime.datetime.strftime(
                        station['lastUpdated'].tz_convert(None), '%Y-%m-%d'
                    ),
                    freq='H'
                )
                user_dates = pd.date_range(
                    pd.to_datetime(self.time_range['start']),
                    pd.to_datetime(self.time_range['end']),
                    freq='H'
                )
                freq_of_user_in_openaq_dates = 0
                for user_date in user_dates:
                    if user_date in openaq_dates:
                        freq_of_user_in_openaq_dates += 1
                is_in_temporal_range.append(
                    freq_of_user_in_openaq_dates / len(user_dates)
                )
        logger.info("All stations have been explored.")
        stations['distance'] = distances
        stations['is_in_temporal_range'] = is_in_temporal_range
        stations = stations[stations['is_in_temporal_range'] != -1]
        stations[
            'combination_dist_and_is_in_temporal_range'
        ] = stations['distance'] / stations['is_in_temporal_range']
        stations = stations.sort_values(
            'combination_dist_and_is_in_temporal_range', ascending=True
        )
        return stations

    def _get_closest_stations_to_location(self) -> pd.DataFrame:
        """
        Method to check which stations are within 100km of the location of
        interest. If there are stations with the 'sensorType' parameter equal
        to 'reference grade' they are chosen over the 'low-cost sensor' class.
        If no stations are retrieved, an exception is thrown.
        """
        # Command to retrieve the stations within 100km for the variable and
        # location of interest
        stations = self.api.locations(
            parameter=self.variable,
            coordinates=f"{self.location.latitude},"
                        f"{self.location.longitude}",
            radius=100000, df=True)

        # Throw an exception if not stations are retrieved
        if len(stations) == 0:
            logger.error("No stations are retrieved.")
            raise Exception('There are no stations next to this location in '
                            'OpenAQ for the variable of interest')

        # Preference of 'reference grade' sensor types over 'low-cost'
        if len(stations[stations['sensorType'] == 'reference grade']) >= 1:
            stations = stations[stations['sensorType'] == 'reference grade']
        return stations

    def get_data(self, stations: pd.DataFrame) -> List[xr.Dataset]:
        stations_downloaded = 0
        xr_datasets = []
        for i, station in stations.iterrows():
            try:
                xr_data = self._get_data(station)
                xr_datasets.append(xr_data)
                stations_downloaded += 1
                if stations_downloaded == 5:
                    break
            except Exception as ex:
                logger.error(ex)
                continue

        if len(xr_datasets) == 0:
            raise Exception('No data was retrieved')

        return xr_datasets

    def _get_data(self, station: pd.Series) -> xr.Dataset:
        """
        This methods retrieves data from the OpenAQ platform in pd.DataFrame 
        format.
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
        xr_data = self.create_xarray_dataset_with_attrs(data, station)
        return xr_data

    def check_variable_in_station(self, station: pd.Series):
        """
        This method checks whether the stations has available data for the
        variable of interest
        """
        if self.variable not in [x['parameter'] for x in station['parameters']]:
            raise Exception('The variable intended to extraction is not'
                            ' available for the nearest / exact location')

    def concat_filter_and_save_data(
            self,
            datasets: List[xr.Dataset],
            output_path_data: Path
    ):
        """
        This function saves the data in netcdf format
        """
        data = xr.concat(datasets, dim='station_id')
        
        if ('o3' in data.data_vars) and (data.o3.attrs['units'] == 'ppm'):
            # https://www.teesing.com/en/page/library/tools/ppm-mg3-converter
            molecular_weight_o3 = 48.00  # g/mol
            data['o3'] *= 0.0409 * molecular_weight_o3  # this is mg/m3
            data['o3'] *= 1e3  # this is ug/m3
            data.o3.attrs['units'] = 'microgram / m^3'
            logger.info(f"O3 has been converted from ppm to micrograms / m^3")
            
        if ('no2' in data.data_vars) and (data.no2.attrs['units'] == 'ppm'):
            # https://www.teesing.com/en/page/library/tools/ppm-mg3-converter
            molecular_weight_no2 = 46.01  # g/mol
            data['no2'] *= 0.0409 * molecular_weight_no2  # this is mg/m3
            data['no2'] *= 1e3  # this is ug/m3
            data.no2.attrs['units'] = 'microgram / m^3'
            logger.info(f"NO2 has been converted from ppm to micrograms / m^3")

        if ('pm25' in data.data_vars) and (data.pm25.attrs['units'] == 'ppm'):
            raise ValueError("The observed pm25 are in ppm.")
        
        utils.write_netcdf(output_path_data, data)

    def create_xarray_dataset_with_attrs(self,
                                         data: pd.DataFrame,
                                         station: pd.Series) -> xr.Dataset:
        xr_ds = data['value'].to_xarray().rename(
            {'date.utc': 'time'}
        ).to_dataset(
            name=self.variable
        )
        xr_ds['time'] = pd.to_datetime(xr_ds.time.values)
        xr_ds['station_id'] = station['id']
        xr_ds = xr_ds.set_coords('station_id')
        xr_ds = xr_ds.expand_dims('station_id')
        xr_ds['x'] = station['coordinates.longitude']
        xr_ds = xr_ds.set_coords('x')
        xr_ds['y'] = station['coordinates.latitude']
        xr_ds = xr_ds.set_coords('y')
        xr_ds['_x'] = self.location.longitude
        xr_ds = xr_ds.set_coords('_x')
        xr_ds['_y'] = self.location.latitude
        xr_ds = xr_ds.set_coords('_y')
        xr_ds['distance'] = station['distance']
        xr_ds = xr_ds.set_coords('distance')

        xr_ds.y.attrs['units'] = 'degrees_north'
        xr_ds.y.attrs['long_name'] = 'Latitude'
        xr_ds.y.attrs['standard_name'] = 'latitude'

        xr_ds.x.attrs['units'] = 'degrees_east'
        xr_ds.x.attrs['long_name'] = 'Longitude'
        xr_ds.x.attrs['standard_name'] = 'longitude'

        xr_ds.distance.attrs['units'] = 'km'
        xr_ds.distance.attrs['long_name'] = 'Distance'
        xr_ds.distance.attrs['standard_name'] = 'distance'

        xr_ds['_y'].attrs['units'] = 'degrees_north'
        xr_ds['_y'].attrs['long_name'] = 'Latitude of the location of interest'
        xr_ds['_y'].attrs['standard_name'] = 'latitude_interest'

        xr_ds['_x'].attrs['units'] = 'degrees_east'
        xr_ds['_x'].attrs['long_name'] = 'Longitude of the location of interest'
        xr_ds['_x'].attrs['standard_name'] = 'longitude_interest'

        xr_ds.station_id.attrs['long_name'] = 'station name'
        xr_ds.station_id.attrs['cf_role'] = 'timeseries_id'
        
        units = data.unit.unique()
        if len(units) == 1:
            if isinstance(units[0], bytes): units[0] = units[0].decode("utf-8")
            xr_ds[self.variable].attrs['units'] = units2str[units[0]]
            xr_ds[self.variable].attrs['standard_name'] = self.variable
            xr_ds[self.variable].attrs['long_name'] = var2longstr[self.variable]
            logger.info(f"Dataset of {self.variable} is in {units[0]}.")
        else:
            logger.error(f"Units of {self.variable} observations downloaded "
                         f"from OpenAQ are not homgeneous")
            raise ValueError(f"Observations extracted are not homgeneous.")

        xr_ds.attrs['featureType'] = "timeSeries"
        xr_ds.attrs['Conventions'] = "CF-1.4"
        return xr_ds


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
