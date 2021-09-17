import datetime
import glob
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from src.data.observations import OpenAQDownloader
from src.data.utils import Location
from src.constants import ROOT_DIR
from src.workflow import Workflow
from src.data.transformer import LocationTransformer
from pathlib import Path

variable = "o3"
station_id = "GB002"

# Get CAMS forecast and Corrected CAMS Forecast into predictions dataframe
for day in range(1, 32):
    time_0 = datetime.datetime.utcnow()
    w = Workflow(
        variable=variable,
        date=datetime.datetime(year=2021, month=8, day=day),
        model=Path("/home/pereza/datos/cams") / "config_inceptiontime_depth6.yml",
        data_dir=Path("/home/pereza/datos/cams"),
        stations_csv=ROOT_DIR / "data/external/stations.csv",
        station_id=station_id,
    )
    w.run()
    time_1 = datetime.datetime.utcnow()
    total_time = time_1 - time_0
    print(total_time.total_seconds())

files = glob.glob(f'/home/pereza/datos/cams/*/{variable}_*_{station_id}_*.csv')
predictions = []
for file in files:
    predictions.append(pd.read_csv(file, index_col=0))
predictions_df = pd.concat(predictions)
predictions_df.index = pd.to_datetime(predictions_df.index)
predictions_df = predictions_df.sort_index()

# Get observations for the same time period into obs dataframe
stations = pd.read_csv(
    ROOT_DIR / "data" / "external" / "stations.csv",
    index_col=0,
    names=[
        "location_id",
        "city",
        "country",
        "latitude",
        "longitude",
        "timezone",
        "elevation",
    ],
)
station = stations[stations["location_id"] == station_id]
dict_to_location = station.iloc[0].to_dict()
for var in ["longitude", "latitude", "elevation"]:
    dict_to_location[var] = float(dict_to_location[var])
location_obj = Location(**dict_to_location)

obs_path = OpenAQDownloader(
    location_obj,
    Path('/home/pereza/datos/cams'),
    variable,
    dict(start='2021-08-01', end='2021-08-31')
).run()

obs = xr.open_dataset(obs_path)
obs = obs.resample({"time": "1H"}).mean()
obs = LocationTransformer(
    variable=variable, location=location_obj
).weight_average_with_distance(obs)
for data_var in list(obs.data_vars.keys()):
    obs = obs.rename(
        {data_var: f"{data_var}_observed"}
    )
obs = LocationTransformer(
    variable=variable, location=location_obj
).filter_observations_data(obs)
obs = obs.resample({"time": "1H"}).asfreq()
for data_var in list(obs.data_vars.keys()):
    obs[data_var] = obs[data_var].interpolate_na(
        dim="time",
        method="linear",
        fill_value="extrapolate",
        use_coordinate=True,
        max_gap=pd.Timedelta(value=12, unit="h")
    )
obs = obs.to_dataframe()
obs = obs.rename(columns={f'{variable}_observed': 'Observations'})
# Join both Predictions and Observations into a single dataframe which is used to plot
data = predictions_df.join([obs])
pearson_corr = {
    'CAMS': np.round(np.corrcoef(
        data.dropna()['CAMS'].values, data.dropna()['Observations'].values
    )[0][1], 2),
    'Corrected CAMS': np.round(np.corrcoef(
        data.dropna()['Corrected CAMS'].values, data.dropna()['Observations'].values
    )[0][1], 2)
}
mae_metric = {
    'CAMS': np.round(np.mean(
        np.abs(
            data.dropna()['CAMS'].values - data.dropna()['Observations'].values
        )
    ), 2),
    'Corrected CAMS': np.round(np.mean(
        np.abs(
            data.dropna()['Corrected CAMS'].values - data.dropna()['Observations'].values
        )
    ), 2)
}
colors = {
    'Observations': "k",
    'CAMS': "red",
    'Corrected CAMS': "green"
}
fig, ax = plt.subplots(figsize=(30, 15))
for i, column in enumerate(data.columns):
    plt.plot(
        data.index.values,
        data[column].values,
        linewidth=3,
        color=colors[column],
        label=column,
    )
    plt.axhline(y=np.mean(data.dropna()[column].values),
                color=colors[column],
                linestyle='-',
                linewidth=3,
                alpha=0.4)
    plt.text(1.02,
             np.mean(data.dropna()[column].values),
             f"{column} mean",
             transform=ax.transAxes,
             fontsize=8,
             color=colors[column])
textstr = '\n'.join((
    f'CAMS Pearson Correlation = {pearson_corr["CAMS"]}',
    f'Corrected CAMS Pearson Correlation = {pearson_corr["Corrected CAMS"]}',
    f'CAMS MAE = {mae_metric["CAMS"]}',
    f'Corrected CAMS MAE = {mae_metric["Corrected CAMS"]}'
))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.01, 0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props)
plt.legend()
plt.ylabel(variable + r" ($\mu g / m^3$)", fontsize="xx-large")
plt.xlabel("Date", fontsize="x-large")
plt.title(
    f"{location_obj.city} ({location_obj.country})", fontsize="xx-large"
)
plt.show()
