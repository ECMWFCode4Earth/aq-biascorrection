import logging
import os
from math import pi
from pathlib import Path
from typing import List, NoReturn

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import constants

log = logging.getLogger("Station Plotter")
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
        """ Class that handles the visualization generation for all data at each
        station.

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
        self.country = country
        st_metadata = pd.read_csv(metadata_path)
        
        # Load stations data
        self.sts_df = st_metadata[st_metadata.country == country]
        if stations is not None:
            self.sts_df = self.sts_df[self.sts_df.city.isin(stations)]
        ids = self.sts_df.id.values
        paths = [data_path / varname / f"data_{varname}_{id}.csv" for id in ids]
        self.data = {}
        self.codes = []
        for i, path in enumerate(paths):
            if os.path.exists(path):
                log.debug(f"Data for station {ids[i]} is found.")
                self.data[ids[i]] = pd.read_csv(path, index_col=0)
                self.codes.append(ids[i])
            else:
                log.info(f"Data for station {ids[i]} is not found.")

    def plot_data(self, output_path: Path = None, agg: str = None) -> NoReturn:
        """ Plot the for the variable requested in the stations whose position 
        is specified.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            log.debug(f"Plotting data for {info.city.values[0]}")
            df = self.data[st_code].set_index('index')
            df.index.name = 'Date'
            if agg: 
                df = aggregate_df(df, agg)
            df[f'{self.varname}_forecast'] = df[f'{self.varname}_observed'] + \
                df[f'{self.varname}_bias']
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d %HH')

            df[[f'{self.varname}_forecast',
                f'{self.varname}_observed']].plot(figsize=(23, 12))
            plt.legend(["Forecast", "Observed"], title=self.varname.upper(), 
                       fontsize='x-large', title_fontsize='x-large')
            plt.title(f"{info.city.values[0]} ({info.country.values[0]})", 
                      fontsize='xx-large')
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            plt.xlabel("Date", fontsize='x-large')
            if agg: 
                x_pos = float(np.mean(plt.gca().get_xlim()))
                y_pos = float(np.average(plt.gca().get_ylim(), 
                                         weights=[0.05, 0.95]))
                annot = agg.capitalize() + " aggregated data"
                plt.text(x_pos, y_pos, annot, fontsize='x-large', ha='center')
            plt.tight_layout()

            if output_path:
                city = ''.join(info.city.values[0].split(' ')).lower()
                country = ''.join(info.country.values[0].split(' ')).lower()
                freq = f'{agg}_' if agg else ''
                filename = f"{freq}{self.varname}_bias_{city}_{country}.png"
                output_filename = output_path / filename
                log.info(f"Plot saved to {output_filename}.")
                plt.savefig(output_filename)
        if not output_path:
            plt.show()

    def plot_correlations(
        self, 
        output_path: Path = None,
        agg: str = None
    ) -> NoReturn:
        """ Plort the correlation between the prediction bias and the model
        features.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            log.debug(f"Plotting data for {info.city.values[0]}")
            df = self.data[st_code].set_index('index')
            df[f'{self.varname}_forecast'] = df[f'{self.varname}_observed'] + \
                df[f'{self.varname}_bias']
            df = df.drop(f'{self.varname}_observed', axis=1)
            df['local_time_hour'] = np.cos(2 * pi * df['local_time_hour'] / 24)\
                + np.sin(2 * pi * df['local_time_hour'] / 24)

            if agg:
                df = df.drop('local_time_hour', axis=1)
                df = aggregate_df(df, agg)

            df = df.rename({f'{self.varname}_bias' : f'{self.varname} Bias',
                            'local_time_hour': 'Local time'}, axis=1)
            df.columns = [col.split('_')[0].upper() for col in df.columns]
            plt.figure(figsize=(26, 14))
            mask = np.triu(np.ones(df.shape[1], dtype=np.bool))
            ax = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='RdBu', mask=mask, 
                             annot=True)
            plt.setp(ax.get_yticklabels()[0], visible=False)  
            plt.setp(ax.get_xticklabels()[-1], visible=False)
            xticks = ax.xaxis.get_major_ticks()
            xticks[-1].set_visible(False)
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].set_visible(False)
            plt.title(f"{info.city.values[0]} ({info.country.values[0]})", 
                      fontsize='xx-large')
            if agg: 
                x_pos = len(df.columns) / 2
                y_pos = 1
                annot = agg.capitalize() + " aggregated data"
                plt.text(x_pos, y_pos, annot, fontsize='large', ha='center')
            plt.xticks(rotation=65, fontsize='x-large')
            plt.yticks(rotation=0, fontsize='x-large')

            if output_path:
                city = ''.join(info.city.values[0].split(' ')).lower()
                country = ''.join(info.country.values[0].split(' ')).lower()
                filename = f"{agg + '_' if agg else ''}corrs_{self.varname}" \
                           f"_bias_{city}_{country}.png"
                output_filename = output_path / filename
                log.info(f"Plot saved to {output_filename}.")
                plt.savefig(output_filename)

        if not output_path:
            plt.show()

    def plot_hourly_bias(
        self, 
        show_std: bool = True, 
        output_path: Path = None
    ) -> NoReturn:
        """ Plot the bias for the variable requested in the stations whose
        position is specified.

        Args:
            show_std (bool): whether to show the empirical standard deviation
            or not. By default, it is shown.
            output_path (Path): the output folder at which save the images.
        """
        stats = ['mean', 'std']
        bias_var = f"{self.varname}_bias"
        means = pd.DataFrame(index=list(range(24)))
        stds = pd.DataFrame(index=list(range(24)))
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            log.debug(f"Plotting data for {info.city.values[0]}")
            data = self.data[st_code]
            agg_h = data.groupby('local_time_hour').agg(stats)[bias_var]
            means[info.city.values[0]] = agg_h['mean']
            stds[info.city.values[0]] = agg_h['std']
            
        if len(means.columns) == 0:
            log.error(f"No data available for any station in {self.country}")
            return None

        m = pd.DataFrame(means, index=agg_h.index)
        s = pd.DataFrame(stds, index=agg_h.index)
        if show_std:
            m.plot.bar(yerr=s, capsize=4, rot=0, figsize=(20, 14))
        else:
            m.plot.bar(figsize=(26, 14))
        plt.axhline(0, ls='--', lw=2, c='k')
        plt.xlabel("Local Time", fontsize='x-large')
        plt.title(f"{self.varname.upper()} bias in {info.country.values[0]}")
        plt.tight_layout()
        plt.legend(title='City', fontsize='x-large', title_fontsize='x-large')
        if output_path:
            country = ''.join(info.country.values[0].split(' ')).lower()
            filename = f"hourly_{self.varname}_bias_{country}.png"
            output_filename = output_path / filename
            log.info(f"Plot saved to {output_filename}.")
            plt.savefig(output_filename)
        else:
            plt.show()

    def plot_bias_cdf(
        self, 
        output_path: str = None, 
        agg: str = None
    ) -> NoReturn:
        """Plot the CDF of bias for the variable requested in the stations whose
        position is specified.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        target = f'{self.varname}_bias'
        dfs = []
        labels = []
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            log.debug(f"Plotting data for {info.city.values[0]}")
            data = self.data[st_code]
            ndays = len(aggregate_df(data, 'daily', 'index').index)
            log.debug(f"There are {len(data.index)} observation which corresponds to a total of {ndays} days.")
            
            if agg: 
                data = aggregate_df(data, agg, 'index')
            data['City'] = f"{info.city.values[0]} ({ndays:.0f})"
            dfs.append(data)
            labels.append(f"{info.city.values[0]} ({ndays:.0f})")
        
        if len(dfs) == 0:
            log.error(f"No data available for any station in {self.country}")
            return None
        df = pd.concat(dfs)
        g = sns.FacetGrid(df, hue="City", height=8, aspect=1.6, legend_out=True)
        g = g.map_dataframe(sns.histplot, target, stat='probability', kde=True, 
                            binwidth=5, legend=True)
        freq = agg + ' ' if agg else ''
        plt.title(f"CDF of {freq}{self.varname.upper()}"
                  f" bias in {info.country.values[0]}")
        plt.legend(labels, title='City (days available)', 
                   title_fontsize='x-large', fontsize='x-large')
        plt.ylabel("Probability", fontsize='x-large')
        plt.xlabel(freq + target.replace("_", " ").capitalize(), 
                   fontsize='x-large')
        plt.tight_layout()
        if output_path:
            country = ''.join(info.country.values[0].split(' ')).lower()
            freq = freq.replace(' ', '_')
            filename = f"{freq}bias_cdf_{self.varname}_bias_{country}.png"
            output_filename = output_path / filename
            log.info(f"Plot saved to {output_filename}.")
            plt.savefig(output_filename)
        else:
            plt.show()


def aggregate_df(df, agg, index_col: str = None) -> pd.DataFrame:
    """ Aggregate values in dataframe aggregating them by a column specified. 
    The column must represent a date.

    Args:
        df (pd.DataFrame): dataframe to aggregate.
        agg (str): period to aggregate the data. Choices are: daily or monthly.
        index_col (str): The column name to use as index for aggregation. 
        Consider the index as default.
    """
    if index_col:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.resample(constants.str2agg[agg], on=index_col)
    else:
        df.index = pd.to_datetime(df.index)
        df = df.resample(constants.str2agg[agg])
    
    return df.mean().dropna()
