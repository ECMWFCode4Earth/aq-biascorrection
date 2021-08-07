import glob
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from ipywidgets import interactive, widgets
from matplotlib.dates import DateFormatter
from pydantic.dataclasses import dataclass

from src.constants import ROOT_DIR
from src.data.utils import Location
from src.logging import get_logger

df_stations = pd.read_csv(ROOT_DIR / "data" / "external" / "stations.csv", index_col=0)
date_form = DateFormatter("%-d %b %y")

logger = get_logger('Validation Visualization')


@dataclass
class ValidationVisualization:
    validation_datasets: list
    varname: str
    location: Location
    class_on_train: str
    output_dir: Path

    def run(self):
        logger.info('Plotting CDF Bias for "Hour", "Day" and "Month" aggregations')
        self.plot_bias_cdf(
            self.validation_datasets,
            self.location,
            'hour'
        )
        self.plot_bias_cdf(
            self.validation_datasets,
            self.location,
            'day'
        )
        self.plot_bias_cdf(
            self.validation_datasets,
            self.location,
            'month'
        )
        logger.info('Plotting "Hour" and "Month" aggregation Box Plots taking into '
                    'consideration training and test datasets')
        if self.class_on_train == 'all':
            self.box_plot_time_agg(
                self.validation_datasets,
                self.location,
                True,
                'hour')
            self.box_plot_time_agg(
                self.validation_datasets,
                self.location,
                True,
                'month')
        self.box_plot_time_agg(
            self.validation_datasets,
            self.location,
            False,
            'hour')
        self.box_plot_time_agg(
            self.validation_datasets,
            self.location,
            False,
            'month')
        logger.info('Plotting "Hour" and "Month" aggregation ErrorBar Plots')
        self.time_serie_time_agg(
            self.validation_datasets,
            self.location,
            'hour')
        self.time_serie_time_agg(
            self.validation_datasets,
            self.location,
            'month')
        logger.info('Plotting TimeSerie Plots')
        self.time_serie_total(
            self.validation_datasets,
            self.location)

    def get_dataset_for_boxplot(
            self,
            initialization_datasets
    ) -> pd.DataFrame:
        """
        Method to join the initialization into one dataset which serves to create
        a boxplot
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
        """
        datasets = []
        for init in initialization_datasets:
            data = init.cams.join([init.observations, init.predictions])
            for column in data.columns:
                data[column] = data[column].astype(float)
            data['class_on_train'] = init.class_on_train
            data = data.set_index([data.index.values, 'class_on_train'])
            data_stck = data.stack().reset_index().set_index(
                'level_0'
            ).rename(columns={'level_2': 'Data Type',
                              'class_on_train': 'Class on train',
                              0: self.varname + r' ($\mu g / m^3$)'})
            datasets.append(data_stck)
        data = pd.concat(datasets)
        data['Data Kind'] = [
            f"{data.iloc[i]['Data Type']} - ({data.iloc[i]['Class on train']})"
            for i in range(len(data))
        ]
        data = data.sort_values(by='Data Kind')
        return data

    @staticmethod
    def get_dataset_for_timeseries(
            initialization_datasets
    ) -> pd.DataFrame:
        """
        Method to join the initialization into one dataset which serves to create
        a time serie
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
        """
        datasets = []
        for init in initialization_datasets:
            data = init.cams.join([init.observations, init.predictions])
            for column in data.columns:
                data[column] = data[column].astype(float)
            datasets.append(data)
        data = pd.concat(datasets)
        return data

    def box_plot_time_agg(
            self,
            initialization_datasets,
            location,
            compare_train_test: bool = True,
            agg_time: str = 'hour',
    ):
        """
        Method to create a boxplot that compares the value of all the different
        initializations passed as an argument.
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
            location: Location object for the station_id wanted
            compare_train_test: whether to use all the data or to differ between train
                                and test initializations
            agg_time: indicates how the boxplot x-axis is shown, i.e. 'hour' or 'month'
        """
        city = "".join(location.city.split(" ")).lower()
        country = "".join(location.country.split(" ")).lower()
        station_code = "".join(location.location_id.split(" ")).lower()
        freq = f"{agg_time}_" if agg_time else ""
        compare = 'train-test-comparison_' if compare_train_test else ""
        filename = f"{freq}{compare}{self.varname}_boxplot_" \
                   f"{station_code}_{city}_{country}.png"
        plot_path = self.output_dir / 'BoxPlot' / filename
        if not plot_path.parent.exists():
            os.makedirs(plot_path.parent, exist_ok=True)
        if plot_path.exists():
            return None
        data = self.get_dataset_for_boxplot(initialization_datasets)
        if agg_time == 'hour':
            x_data = data.index.hour
        elif agg_time == 'month':
            x_data = data.index.month
        else:
            raise NotImplementedError(
                'There are only two aggregation types for time: "hour" and "month".'
            )
        plt.figure(figsize=(30, 15))
        if compare_train_test:
            sns.boxplot(x=x_data,
                        y=self.varname + r' ($\mu g / m^3$)',
                        hue='Data Kind',
                        palette=["b", "g", "orange", 'red', 'm', 'yellow'],
                        data=data)
        else:
            sns.boxplot(x=x_data,
                        y=self.varname + r' ($\mu g / m^3$)',
                        hue='Data Type',
                        palette=["b", "g", "orange", 'red', 'm', 'yellow'],
                        data=data)
        plt.ylabel(self.varname + r' ($\mu g / m^3$)', fontsize='xx-large')
        plt.xlabel(f"{agg_time.capitalize()} of prediction", fontsize='xx-large')
        plt.title(f"{location.city} ({location.country})", fontsize='xx-large')
        plt.savefig(plot_path,
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    def time_serie_time_agg(
            self,
            initialization_datasets: list,
            location: Location,
            agg_time: str = 'hour'
    ):
        """
        Method to create a scatter plot that compares the value of all the different
        initializations passed as an argument, showing mean and std (errorbar).
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
            location: Location object for the station_id wanted
            agg_time: indicates how the boxplot x-axis is shown, i.e. 'hour' or 'month'
        """
        city = "".join(location.city.split(" ")).lower()
        country = "".join(location.country.split(" ")).lower()
        station_code = "".join(location.location_id.split(" ")).lower()
        freq = f"{agg_time}_" if agg_time else ""
        filename = f"{freq}{self.varname}_errorbarplot_" \
                   f"{station_code}_{city}_{country}.png"
        plot_path = self.output_dir / 'ErrorBarPlot' / filename
        if not plot_path.parent.exists():
            os.makedirs(plot_path.parent, exist_ok=True)
        if plot_path.exists():
            return None
        colors = ['b', 'k', 'orange']
        data = self.get_dataset_for_timeseries(initialization_datasets)
        if agg_time == 'hour':
            x_data = data.index.hour
        elif agg_time == 'month':
            x_data = data.index.month
        else:
            raise NotImplementedError(
                'There are only two aggregation types for time: "hour" and "month".'
            )
        plt.figure(figsize=(17, 9))
        for i, column in enumerate(data.columns):
            plt.errorbar(data[column].groupby(x_data).mean().index.values,
                         data[column].groupby(x_data).mean().values,
                         data[column].groupby(x_data).std().values,
                         label=column,
                         marker='o',
                         color=colors[i],
                         ecolor=colors[i],
                         ls='',
                         alpha=0.5)
        plt.legend()
        plt.ylabel(self.varname + r' ($\mu g / m^3$)', fontsize='xx-large')
        plt.xlabel(f"{agg_time.capitalize()} of prediction", fontsize='xx-large')
        plt.title(f"{location.city} ({location.country})", fontsize='xx-large')
        plt.savefig(plot_path,
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    def time_serie_total(
            self,
            initialization_datasets,
            location,
            xlim=None
    ):
        """
        Method to plot the CDF of bias for the CAMS forecast and
        the Corrected CAMS forecast in a specific location.
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
            location: Location object for the station_id wanted
            xlim: indicates how the boxplot x-axis is shown, i.e. 'hour' or 'month'
        """
        city = "".join(location.city.split(" ")).lower()
        country = "".join(location.country.split(" ")).lower()
        station_code = "".join(location.location_id.split(" ")).lower()
        filename = f"{self.varname}_timeserie_" \
                   f"{station_code}_{city}_{country}.png"
        plot_path = self.output_dir / 'TimeSeriePlot' / filename
        if not plot_path.parent.exists():
            os.makedirs(plot_path.parent, exist_ok=True)
        if plot_path.exists():
            return None
        colors = ['b', 'k', 'orange']
        data = self.get_dataset_for_timeseries(initialization_datasets)
        plt.figure(figsize=(17, 9))
        for i, column in enumerate(data.columns):
            plt.plot(
                data.index.values,
                data[column].values,
                linewidth=2,
                linestyle='dashed',
                color=colors[i],
                label=column
            )
        plt.legend()
        plt.ylabel(self.varname + r' ($\mu g / m^3$)', fontsize='xx-large')
        plt.xlabel("Date", fontsize='xx-large')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(date_form)
        if xlim is not None:
            plt.xlim(xlim)
        plt.title(f"{location.city} ({location.country})", fontsize='xx-large')
        plt.savefig(plot_path,
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_bias_cdf(self,
                      initialization_datasets: list,
                      location: Location,
                      agg_time: str = None):
        """
        Method to plot the CDF of bias for the CAMS forecast and
        the Corrected CAMS forecast in a specific location.
        Args:
            initialization_datasets: list of InitializationDataset with data of CAMS,
                                     Observations, Predictions and its Class during the
                                     training phase.
            location: Location object for the station_id wanted
            agg_time: indicates how the boxplot x-axis is shown, i.e. 'hour' or 'month'
        """
        city = "".join(location.city.split(" ")).lower()
        country = "".join(location.country.split(" ")).lower()
        station_code = "".join(location.location_id.split(" ")).lower()
        freq = f"{agg_time}_" if agg_time else ""
        filename = f"{freq}{self.varname}_cdf-bias_" \
                   f"{station_code}_{city}_{country}.png"
        plot_path = self.output_dir / 'CDFBiasPlot' / filename
        if not plot_path.parent.exists():
            os.makedirs(plot_path.parent, exist_ok=True)
        if plot_path.exists():
            return None
        data = self.get_dataset_for_timeseries(initialization_datasets)
        if agg_time == 'hour':
            agg_time = 'hourly'
            data = data.resample('H').mean()
        elif agg_time == 'day':
            agg_time = 'daily'
            data = data.resample('D').mean()
        elif agg_time == 'month':
            agg_time = 'monthly'
            data = data.resample('M').mean()
        cams_bias = data['CAMS Forecast'] - data['Observations']
        cams_bias = cams_bias.to_frame('CAMS Bias')
        predictions_bias = data['CAMS + Correction'] - data['Observations']
        predictions_bias = predictions_bias.to_frame('CAMS + Correction Bias')
        df = cams_bias.join(predictions_bias)
        df_n = df.stack().reset_index().set_index('level_0').rename(
            columns={'level_1': 'Data', 0: 'value'}
        )
        plt.figure(figsize=(17, 9))
        g = sns.FacetGrid(df_n, hue="Data", height=8, aspect=1.6, legend_out=True)
        g.map_dataframe(
            sns.histplot,
            'value',
            stat="probability",
            kde=True,
            binwidth=5,
            legend=True
        )
        freq_str = agg_time + " " if agg_time else ""
        plt.title(
            f"CDF of {freq_str.capitalize()}{self.varname.upper()}" f" bias in"
            f" {location.city} ({location.country})"
        )
        plt.legend(
            df.columns,
            title="Data",
            title_fontsize="x-large",
            fontsize="x-large",
        )
        plt.ylabel("Probability", fontsize="x-large")
        plt.xlabel(
            freq_str.capitalize() + self.varname.upper() + r" bias ($\mu g / m^3$)",
            fontsize="x-large",
        )
        plt.savefig(plot_path,
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()


# Methods for implementation of Jupyter Tool
def get_all_locations() -> List[str]:
    return list(df_stations.city.unique())


def get_id_location(city: str) -> str:
    return df_stations.loc[df_stations.city == city, "id"].values[0]


def interactive_viz(varname: str, station: str, date_range: tuple):
    plotter = ValidationVisualization("InceptionTime_ensemble", varname)
    plotter.run(get_id_location(station), date_range)
