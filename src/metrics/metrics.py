from pathlib import Path
from src.metrics.utils import weighted_corr

import xarray as xr
import pandas as pd
import numpy as np


class ScoresTables:
    """
    Class to compute and save the tables with the scores:
    NMAE, BIAS, RMSE, PEARSON CORRELATION, Debiased-NMAE
    """
    def __init__(
            self,
            station_path_observations: Path,
            station_path_forecast: Path,
            metrics_output_dir: Path = Path('../../reports/tables/')
    ):
        self.observations = xr.open_dataset(station_path_observations,
                                   chunks={'time': 100})
        self.forecasts = xr.open_dataset(station_path_forecast,
                                         chunks={'time':100})
        params = station_path_observations.name.split('_')
        self.location_parameters = {}
        for i, param in enumerate(['variable', 'country', 'city',
                                   'location_id', 'time_range']):
            self.location_parameters[param] = params[i]
        self.metrics_output_dir = metrics_output_dir

    def run(self):
        diff = self.forecasts - self.observations
        tables = []
        tables.append(self.nmae_tables(diff, self.observations))
        tables.append(self.bias_tables(diff))
        tables.append(self.rmse_tables(diff))
        tables.append(self.nmae_debiased(self.forecasts, self.observations))
        tables.append(self.pearson_tables(self.forecasts, self.observations))
        table_merged = self.merging_tables(tables)
        opath = self.get_output_path()
        print("Writing table to {}".format(opath))
        table_merged.to_excel(opath,
                              float_format='%.2f')

    def nmae_tables(self, diff: xr.Dataset, obs: xr.Dataset):
        varname = self.location_parameters['variable']
        abs_diff = np.abs(diff)
        mae = abs_diff.mean(dim="time")[varname]
        nmae = mae / obs.mean(dime="time")
        # Table by study case, station and physics
        nmae_by_station_code = nmae.to_pandas().T
        nmae_by_station_code.loc["average"] = nmae_by_station_code.mean(axis=0)
        return nmae_by_station_code

    def bias_tables(self, diff: xr.Dataset):
        varname = self.location_parameters['variable']
        bias = diff.mean(dim='time').mean(dim='study_case')[varname]
        bias_by_station_code = bias.to_pandas().T
        bias_by_station_code.loc["average"] = bias_by_station_code.mean(axis=0)
        return bias_by_station_code

    def nmae_debiased(self, forecast: xr.Dataset, observation: xr.Dataset):
        diff = forecast - observation
        bias = diff.mean(dim='time').mean(dim='study_case')
        debiased_forecast = forecast - bias
        diff = debiased_forecast - observation
        return self.nmae_tables(diff, observation)

    def rmse_tables(self, diff: xr.Dataset):
        varname = self.location_parameters['variable']
        rmse_by_station_code = np.sqrt((diff ** 2).mean(dim='time')[varname])
        rmse_by_station_code = rmse_by_station_code.mean(dim='study_case').to_pandas().T
        rmse_by_station_code.loc['average'] = rmse_by_station_code.mean(axis=0)
        return rmse_by_station_code

    def pearson_tables(self, forecast: xr.Dataset, observation: xr.Dataset):
        varname = self.location_parameters['variable']
        pearson_by_station_code = weighted_corr(forecast[varname], observation[varname],
                                                dim='time', weights=None, return_p=False)
        pearson_by_station_code = pearson_by_station_code.to_pandas().T
        pearson_by_station_code.loc['average'] = pearson_by_station_code.mean(axis=0)
        return pearson_by_station_code

    def merging_tables(self, tables):
        table_merged = pd.concat(tables, axis=1,
                                 keys=("NMAE", "Bias", "RMSE",
                                       'De-Biased NMAE', 'Pearson Correlation'))
        return table_merged

    def get_output_path(self):
        opath = Path(self.metrics_output_dir,
                     self.location_parameters['country'],
                     self.location_parameters['city'],
                     self.location_parameters['location_id'],
                     self.location_parameters['variable'],
                     f"{self.location_parameters['variable']}_"
                     f"{self.location_parameters['country']}_"
                     f"{self.location_parameters['city']}_"
                     f"{self.location_parameters['location_id']}_"
                     f"{self.location_parameters['time_range']}_metrics.xlsx")
        return opath