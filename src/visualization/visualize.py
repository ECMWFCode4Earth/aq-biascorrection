import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from typing import List, NoReturn


preprocess = lambda ds: ds.expand_dims(['station_id', 'latitude', 'longitude'])


class StationTemporalSeriesPlotter:
    def __init__(
            self,
            varname: str,
            country: str,
            data_path: Path,
            metadata_path: Path = Path("data/external/stations.csv"), 
            stations: List[str] = None
    ):
        """ 

        Args:
            varname (str): variable to consider.
            country (str): Country to select.
            station_path_observations (Path): path to the folder containing the 
            observations.
            station_path_forecasts (Path): path to the folder containing the 
            forecasts. Defaults to None.
            stations (str): Stations of the country to select. Defaults to None,
            which takes all the stations.
        """
        self.varname = varname
        st_metadata = pd.read_csv(metadata_path)
        
        # Load stations data
        self.sts_df = st_metadata[st_metadata.country == country]
        if stations is not None:
            self.sts_df = self.sts_df[self.sts_df.city.isin(stations)]
        ids = self.sts_df.id.values
        paths = [data_path / varname / f"data_{varname}_{id}.csv" for id in ids]
        csvs = [pd.read_csv(path, index_col=0) for path in paths]
        
        self.codes = ids
        self.data = dict(zip(ids, csvs))

    def plot_data(self) -> NoReturn:
        """ Plot the for the variable requested in the stations whose position 
        is specified.

        Args:
            station (int): Position of the station of the country. Defaults to 
            the first station of the country.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            df = self.data[st_code].set_index('index')
            df.index.name = 'Date'
            df[f'{self.varname}_forecast'] = df[f'{self.varname}_observed'] + \
                df[f'{self.varname}_bias']

            df[[f'{self.varname}_forecast', f'{self.varname}_observed']].plot()
            plt.legend(["Forecast", "Observed"], title=self.varname.upper(), 
                    fontsize='large', title_fontsize='large')
            plt.title(f"{info.city.values[0]} ({info.country.values[0]})")
            plt.tight_layout()

        plt.show()

    def plot_correlations(self) -> NoReturn:
        """ Plort the correlation between the prediction bias and the model
        features.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            df = self.data[st_code].set_index('index')
            df[f'{self.varname}_forecast'] = df[f'{self.varname}_observed'] + \
                df[f'{self.varname}_bias']
            df = df.drop(f'{self.varname}_observed', axis=1)
            df = df.rename({f'{self.varname}_bias' : f'{self.varname} Bias',
                            'local_time_hour': 'Local time'}, axis=1)
            df.columns = [col.split('_')[0].upper() for col in df.columns]
            plt.figure()
            sns.heatmap(df.corr(), vmin=-1, vmax=1)
            plt.title(f"{info.city.values[0]} ({info.country.values[0]})", 
                      fontsize='large')
            plt.xticks(rotation=65)
            
        plt.show()    
