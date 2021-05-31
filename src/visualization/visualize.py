import matplotlib.pyplot as plt
import xarray as xr

from pathlib import Path


class StationTemporalSeriesPlotter:
    def __init__(
            self,
            station_path_observations: Path,
            station_path_forecasts: Path = None
    ):
        obs = xr.open_dataset(station_path_observations,
                              chunks={'time': 100})
        if station_path_forecasts is not None:
            forecasts = xr.open_dataset(station_path_forecasts,
                                        chunks={'time':100})
        else:
            forecasts = None

