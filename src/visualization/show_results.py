import os
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

    def load_obs_preds(self, station: str):
        # Load Observations
        idir = ROOT_DIR / "data" / "processed"
        data_file = list(idir.rglob(f"data_{self.varname}_{station}.csv"))[0]
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        return data

    def run(self, station_code: str):
        df = self.load_data(station_code)
        data = self.load_obs_preds(station_code)


if __name__ == '__main__':
    ResultsPlotter("InceptionTime_ensemble", "pm25").run("GB002")
