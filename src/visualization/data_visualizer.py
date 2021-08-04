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
from src.logging import get_logger

logger = get_logger("Station Plotter")
preprocess = lambda ds: ds.expand_dims(["station_id", "latitude", "longitude"])


class StationTemporalSeriesPlotter:
    def __init__(
        self,
        varname: str,
        country: str,
        data_path: Path = Path("data/processed"),
        metadata_path: Path = Path("data/external/stations.csv"),
        stations: List[str] = None,
    ):
        """Class that handles the visualization generation for all data at each
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
                logger.debug(f"Data for station {ids[i]} is found.")
                self.data[ids[i]] = pd.read_csv(path, index_col=0)
                self.codes.append(ids[i])
            else:
                logger.info(f"Data for station {ids[i]} is not found.")

    def plot_data(self, output_path: Path = None, agg: str = None) -> NoReturn:
        """Plot the for the variable requested in the stations whose position
        is specified.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            logger.debug(f"Plotting data for {info.city.values[0]}")
            df = self.data[st_code].set_index('index')
            df.index.name = 'Date'
            if agg: 
                df = aggregate_df(df, agg)
            df[f"{self.varname}_forecast"] = (
                df[f"{self.varname}_observed"] + df[f"{self.varname}_bias"]
            )
            df.index = pd.to_datetime(df.index).strftime("%d %b, %Y")

            df[[f"{self.varname}_forecast", f"{self.varname}_observed"]].plot(
                figsize=(12, 8)
            )
            plt.legend(
                ["Forecast", "Observed"],
                title=self.varname.upper(),
                fontsize="x-large",
                title_fontsize="x-large",
            )
            plt.title(
                f"{info.city.values[0]} ({info.country.values[0]})", fontsize="xx-large"
            )
            plt.xticks(fontsize="x-large")
            plt.yticks(fontsize="x-large")
            plt.ylabel(
                f"{self.varname.upper()} " + r"($\mu g / m^3$)", fontsize="x-large"
            )
            plt.xlabel("Date", fontsize="x-large")
            if agg:
                x_pos = float(np.mean(plt.gca().get_xlim()))
                y_pos = float(np.average(plt.gca().get_ylim(), weights=[0.05, 0.95]))
                annot = agg.capitalize() + " aggregated data"
                plt.text(x_pos, y_pos, annot, fontsize="x-large", ha="center")
            plt.tight_layout()

            if output_path:
                city = "".join(info.city.values[0].split(" ")).lower()
                country = "".join(info.country.values[0].split(" ")).lower()
                freq = f"{agg}_" if agg else ""
                filename = f"{freq}{self.varname}_bias_{city}_{country}.png"
                output_filename = output_path / filename
                logger.info(f"Plot saved to {output_filename}.")
                plt.savefig(output_filename)
        if not output_path:
            plt.show()

    def plot_monthly_bias(self, output_path: Path = None) -> NoReturn:
        """Plot the bias for the variable requested in the stations whose position
        is specified.

        Args:
            output_path (Path): the output folder at which save the images.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            logger.debug(f"Plotting data for {info.city.values[0]}")
            df = self.data[st_code].set_index('index')
            df.index = pd.to_datetime(df.index)
            df_grouped = df[f"{self.varname}_bias"].groupby(df.index.month)
            df = df_grouped.agg(["mean", "std", "count"])
            months = pd.DataFrame(index=list(range(1, 13)))
            df = months.join(df)
            df.index.name = "Date"

            plt.figure(figsize=(10, 7))
            ax = sns.lineplot(data=df.set_index(df.index - 1), y="mean", x="Date")
            ax.fill_between(
                df.index - 1, df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.2
            )
            ax.axhline(0, c="k", ls="--", lw=2)
            ax.tick_params(axis="y", labelcolor="b")
            ax.set_ylabel(
                f"{self.varname.upper()} bias " + r"($\mu g / m^3$)",
                fontsize="x-large",
                color="b",
            )
            plt.yticks(fontsize="x-large")
            plt.xticks(fontsize="x-large")
            ax2 = ax.twinx()
            ax2.set_ylim((0, df["count"].max() * 4))

            df.fillna(0).plot.bar(
                y="count", align="center", ax=ax2, color="red", alpha=0.2
            )
            ax2.set_ylabel("Number observations", color="r", fontsize="x-large")
            ax.legend(
                ["Mean", r"$\pm$ Std"],
                title=self.varname.upper(),
                fontsize="x-large",
                title_fontsize="x-large",
            )
            ax2.legend().set_visible(False)
            plt.title(
                f"{info.city.values[0]} ({info.country.values[0]})", fontsize="xx-large"
            )
            ax2.set_yticks(df["count"].dropna().values)
            min_and_max = df["count"].agg(["max", "min"]).values
            max_count = df["count"].dropna().where(df["count"].isin(min_and_max), "")
            ax2.set_yticklabels(max_count.values)
            ax2.tick_params(axis="y", labelcolor="r")
            ax.set_xlabel("Month", fontsize="x-large")
            plt.tight_layout()

            if output_path:
                city = "".join(info.city.values[0].split(" ")).lower()
                country = "".join(info.country.values[0].split(" ")).lower()
                filename = f"{self.varname}_bias_{city}_{country}.png"
                output_filename = output_path / filename
                logger.info(f"Plot saved to {output_filename}.")
                plt.savefig(output_filename)
        if not output_path:
            plt.show()

    def plot_correlations(self, output_path: Path = None, agg: str = None) -> NoReturn:
        """Plort the correlation between the prediction bias and the model
        features.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            logger.debug(f"Plotting data for {info.city.values[0]}")
            df = self.data[st_code].set_index('index')
            df[f'{self.varname}_forecast'] = df[f'{self.varname}_observed'] + \
                df[f'{self.varname}_bias']
            mean_obs = df[f'{self.varname}_observed'].mean()
            df = df.drop(f'{self.varname}_observed', axis=1)
            df['local_time_hour'] = np.cos(2 * pi * df['local_time_hour'] / 24)\
                + np.sin(2 * pi * df['local_time_hour'] / 24)

            # Deseasonalize the time series
            df = df.drop('local_time_hour', axis=1)
            vars_to_not_deseasonalize = [f'{self.varname} Error Raw']
            df.index = pd.to_datetime(df.index)
            months = df.index.month
            df = df.set_index(months, append=True)
            df[f"{self.varname} Error Raw"] = df[f"{self.varname}_bias"]
            df = df.rename(
                {f"{self.varname}_bias": f"{self.varname} Error\nDeseasonalized"},
                axis=1,
            )
            monthly_mean = df.groupby(months).mean()
            monthly_mean[vars_to_not_deseasonalize] = 0
            df = df.subtract(monthly_mean, level=1)

            if agg:
                df = aggregate_df(df.droplevel(1), agg)

            df.columns = [col.split('_')[0] for col in df.columns]
            plt.figure(figsize=(12, 9))
            mask = np.triu(np.ones(df.shape[1], dtype=np.bool))
            ax = sns.heatmap(
                df.corr().iloc[:, :-2],
                vmin=-1,
                vmax=1,
                cmap="RdBu",
                mask=mask[:, :-2],
                annot=True,
            )
            plt.setp(ax.get_yticklabels()[0], visible=False)
            yticks = ax.yaxis.get_major_ticks()
            yticks[0].set_visible(False)
            plt.title(
                f"{info.city.values[0]} ({info.country.values[0]})", fontsize="xx-large"
            )

            # Print absolute and relative bias values
            abs_bias = -df[f"{self.varname} Error Raw"].mean()
            rel_bias = abs_bias / mean_obs
            x_pos = len(df.columns) / 2
            plt.text(
                x_pos,
                1,
                f"Absolute bias: {abs_bias:.4f}",
                fontsize="large",
                ha="center",
            )
            plt.text(
                x_pos,
                1.8,
                f"Relative bias: {rel_bias:.4f}",
                fontsize="large",
                ha="center",
            )

            if agg:
                annot = agg.capitalize() + " aggregated data"
                plt.text(x_pos, 2.6, annot, fontsize="large", ha="center")

            plt.xticks(rotation=65, fontsize="x-large")
            plt.yticks(rotation=0, fontsize="x-large")
            plt.tight_layout()

            if output_path:
                city = "".join(info.city.values[0].split(" ")).lower()
                country = "".join(info.country.values[0].split(" ")).lower()
                filename = (
                    f"{agg + '_' if agg else ''}corrs_{self.varname}"
                    f"_bias_{city}_{country}.png"
                )
                output_filename = output_path / filename
                logger.info(f"Plot saved to {output_filename}.")
                plt.savefig(output_filename)

        if not output_path:
            plt.show()

    def plot_hourly_bias(
        self, show_std: bool = True, output_path: Path = None
    ) -> NoReturn:
        """Plot the bias for the variable requested in the stations whose
        position is specified.

        Args:
            show_std (bool): whether to show the empirical standard deviation
            or not. By default, it is shown.
            output_path (Path): the output folder at which save the images.
        """
        stats = ["mean", "std"]
        bias_var = f"{self.varname}_bias"
        means = pd.DataFrame(index=list(range(24)))
        stds = pd.DataFrame(index=list(range(24)))
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            logger.debug(f"Plotting data for {info.city.values[0]}")
            data = self.data[st_code]
            agg_h = data.groupby("local_time_hour").agg(stats)[bias_var]
            means[info.city.values[0]] = agg_h["mean"]
            stds[info.city.values[0]] = agg_h["std"]

        if len(means.columns) == 0:
            logger.error(f"No data available for any station in {self.country}")
            return None

        m = pd.DataFrame(means, index=agg_h.index)
        s = pd.DataFrame(stds, index=agg_h.index)
        if show_std:
            m.plot.bar(yerr=s, capsize=4, rot=0, figsize=(10, 7))
        else:
            m.plot.bar(figsize=(10, 7))
        plt.axhline(0, ls="--", lw=2, c="k")
        plt.xlabel("Local Time", fontsize="x-large")
        plt.ylabel(
            f"{self.varname.upper()} bias " + r"($\mu g / m^3$)", fontsize="x-large"
        )
        plt.title(f"CAMS biases in {info.country.values[0]}", fontsize="xx-large")
        plt.tight_layout()
        plt.legend(title="City", fontsize="x-large", title_fontsize="x-large")
        if output_path:
            country = "".join(info.country.values[0].split(" ")).lower()
            filename = f"hourly_{self.varname}_bias_{country}.png"
            output_filename = output_path / filename
            logger.info(f"Plot saved to {output_filename}.")
            plt.savefig(output_filename)
        else:
            plt.show()

    def plot_bias_cdf(self, output_path: str = None, agg: str = None) -> NoReturn:
        """Plot the CDF of bias for the variable requested in the stations whose
        position is specified.

        Args:
            output_path (Path): the output folder at which save the images.
            agg (str): Whether to aggregate data. Choices are: daily or monthly.
        """
        target = f"{self.varname}_bias"
        dfs = []
        labels = []
        for st_code in self.codes:
            info = self.sts_df[self.sts_df.id == st_code]
            logger.debug(f"Plotting data for {info.city.values[0]}")
            data = self.data[st_code]
            ndays = len(aggregate_df(data, 'daily', 'index').index)
            logger.debug(f"There are {len(data.index)} "
                         f"observation which corresponds to a total of {ndays} days.")
            
            if agg: 
                data = aggregate_df(data, agg, 'index')
            data['City'] = f"{info.city.values[0]} ({ndays:.0f})"
            dfs.append(data)
            labels.append(f"{info.city.values[0]} ({ndays:.0f})")

        if len(dfs) == 0:
            logger.error(f"No data available for any station in {self.country}")
            return None
        df = pd.concat(dfs)
        g = sns.FacetGrid(df, hue="City", height=8, aspect=1.6, legend_out=True)
        g = g.map_dataframe(
            sns.histplot, target, stat="probability", kde=True, binwidth=5, legend=True
        )
        freq = agg + " " if agg else ""
        plt.title(
            f"CDF of {freq}{self.varname.upper()}" f" bias in {info.country.values[0]}"
        )
        plt.legend(
            labels,
            title="City (days available)",
            title_fontsize="x-large",
            fontsize="x-large",
        )
        plt.ylabel("Probability", fontsize="x-large")
        plt.xlabel(
            freq.capitalize() + self.varname.upper() + r" bias ($\mu g / m^3$)",
            fontsize="x-large",
        )
        plt.tight_layout()
        if output_path:
            country = "".join(info.country.values[0].split(" ")).lower()
            freq = freq.replace(" ", "_")
            filename = f"{freq}bias_cdf_{self.varname}_bias_{country}.png"
            output_filename = output_path / filename
            logger.info(f"Plot saved to {output_filename}.")
            plt.savefig(output_filename)
        else:
            plt.show()


def aggregate_df(df, agg, index_col: str = None) -> pd.DataFrame:
    """Aggregate values in dataframe aggregating them by a column specified.
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


if __name__ == '__main__':
    StationTemporalSeriesPlotter('pm25', 'Spain').plot_correlations()