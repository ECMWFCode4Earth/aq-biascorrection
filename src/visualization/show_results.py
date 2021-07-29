import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import glob

from ipywidgets import interactive, widgets
from src.constants import ROOT_DIR
from matplotlib.dates import DateFormatter
from typing import List, Dict
from pydantic.dataclasses import dataclass


df_stations = pd.read_csv(ROOT_DIR / "data" / "external" / "stations.csv", index_col=0)
date_form = DateFormatter("%-d %b %y")


@dataclass
class ResultsPlotter:
    model_name: str
    varname: str

    def load_data(self, station: str) -> pd.DataFrame:
        directory = ROOT_DIR / "models" / "results" / self.model_name / self.varname
        sum_dfs = {"train": None,
                   "test": None}
        for key in sum_dfs.keys():
            sum_df, count = None, 0
            for file in glob.glob(str(directory / f"*_{key}.csv")):
                count += 1
                df = pd.read_csv(file, index_col=[0, 1])
                if sum_df is not None:
                    sum_df += df
                else:
                    sum_df = df
            sum_df /= count
            sum_dfs[key] = sum_df.loc[(slice(None), station), :].droplevel(1)
        df_total = pd.concat([sum_dfs['train'], sum_dfs['test']])
        df_total = df_total.sort_index()
        return df_total

    def load_obs_preds(self, station: str) -> pd.DataFrame:
        # Load Observations
        idir = ROOT_DIR / "data" / "processed"
        data_file = list(idir.rglob(f"data_{self.varname}_{station}.csv"))[0]
        data = pd.read_csv(data_file, index_col=0)
        data['index'] = pd.to_datetime(data['index'])
        return data.set_index('index')

    def run(self, station_code: str, xlim: tuple = None):
        df = self.load_data(station_code)
        data = self.load_obs_preds(station_code)

        self.time_serie(df, data, xlim)

    def time_serie(self, df, data, xlim):
        plt.figure(figsize=(17, 9))
        data[f"{self.varname}_forecast"].plot(color='b', label='CAMS forecast')
        data[f"{self.varname}_observed"].plot(color='k', label='Observation')

        is_first = True
        for init_time, values in df.iterrows():
            indices = pd.date_range(start=init_time, periods=len(values), freq='H')
            (data.loc[indices, "pm25_forecast"] - values.values).plot(
                color='orange',
                label="CAMS + Correction" if is_first else '_nolegend_'
            )
            is_first = None
        plt.legend()
        plt.ylabel(self.varname + r' ($\mu g / m^2$)', fontsize='xx-large')
        plt.xlabel("Date", fontsize='xx-large')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(date_form)
        if xlim is not None:
            plt.xlim(xlim)
        plt.show()


# Methods for implementation of Jupyter Tool
def get_all_locations() -> List[str]:
    return list(df_stations.city.unique())


def get_id_location(city: str) -> str:
    return df_stations.loc[df_stations.city == city, "id"].values[0]


def interactive_viz(varname: str, station: str, date_range: tuple):
    plotter = ResultsPlotter("InceptionTime_ensemble", varname)
    plotter.run(get_id_location(station), date_range)


if __name__ == '__main__':
    ResultsPlotter('InceptionTime_ensemble', 'pm25').run("ES001")
