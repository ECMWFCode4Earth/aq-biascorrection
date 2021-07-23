import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from ipywidgets import interactive, widgets
from src.constants import ROOT_DIR
from matplotlib.dates import DateFormatter
from typing import List
from pydantic.dataclasses import dataclass


df_stations = pd.read_csv(ROOT_DIR / "data" / "external" / "stations.csv", index_col=0)
date_form = DateFormatter("%-d %b %y")


@dataclass
class ResultsPlotter:
    model_name: str
    varname: str

    def load_data(self, station: str) -> pd.DataFrame:
        directory = ROOT_DIR / "data" / "predictions" / self.model_name / self.varname
        
        sum_df, count = None, 0
        for file in os.listdir(directory):
            count += 1
            df = pd.read_csv(directory / file, index_col=[0, 1])
            if sum_df is not None:
                sum_df += df
            else:
                sum_df = df
        sum_df /= count
        return sum_df.loc[(slice(None), station), :].droplevel(1)

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
        plt.ylabel(self.varname + r' ($\mu g / m^2$)')
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
    ResultsPlotter('InceptionTime_ensemble', 'pm25').run("GB002")
