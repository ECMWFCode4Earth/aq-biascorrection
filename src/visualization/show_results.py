import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from src.constants import ROOT_DIR
from pydantic.dataclasses import dataclass


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
        data['index'] = pd.to_datetime(data.index)
        return data.set_index('index')

    def run(self, station_code: str):
        df = self.load_data(station_code)
        data = self.load_obs_preds(station_code)

        data[f"{self.varname}_forecast"].plot(color='b', label='CAMS forecast')
        data[f"{self.varname}_observed"].plot(color='k', label='Observation')
        for init_time, values in df.iterrows():
            # TODO: These are corrections so must be joined with predicitons.
            indices = pd.date_range(start=init_time, periods=len(values), freq='H')
            plt.plot(indices, values.values, color='orange')
        plt.show()



if __name__ == '__main__':
    ResultsPlotter("InceptionTime_ensemble", "pm25").run("GB002")
