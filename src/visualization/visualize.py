import matplotlib.pyplot as plt
import xarray as xr

from pathlib import Path


class StationTemporalSeriesPlotter:
    def __init__(
            self,
            station_path_observations: Path,
            station_path_forecasts: Path = None
    ):
        self.obs = xr.open_dataset(station_path_observations,
                                   chunks={'time': 100})
        if station_path_forecasts is not None:
            self.forecasts = xr.open_dataset(station_path_forecasts,
                                             chunks={'time':100})
        else:
            self.forecasts = None
        params = station_path_observations.name.split('_')
        self.location_parameters = {}
        for i, param in enumerate(['variable', 'country', 'city',
                                   'location_id', 'time_range']):
            self.location_parameters[param] = params[i]



