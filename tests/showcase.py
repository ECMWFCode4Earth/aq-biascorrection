import datetime
import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from src.data.observations import OpenAQDownloader
from src.data.utils import Location
from src.constants import ROOT_DIR
from src.workflow import Workflow
from pathlib import Path

variable = "no2"
station_id = "US007"

# Get CAMS forecast and Corrected CAMS Forecast into predictions dataframe
for day in range(1, 32):
    time_0 = datetime.datetime.utcnow()
    w = Workflow(
        variable=variable,
        date=datetime.datetime(year=2021, month=8, day=day),
        model=Path("/home/pereza/datos/cams") / "config_inceptiontime_depth6.yml",
        data_dir=Path("/home/pereza/datos/cams"),
        stations_csv=ROOT_DIR / "data/external/stations.csv",
        station_id="US007",
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
station = stations[stations["location_id"] == "US007"]
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
obs = obs.mean('station_id').to_dataframe()
obs = obs.rename(columns={variable: 'Observations'})
del obs['_x']
del obs['_y']
obs.index = pd.to_datetime(obs.index)
obs = obs.resample('1H').mean()
obs = obs.sort_index()

# Join both Predictions and Observations into a single dataframe which is used to plot
data = predictions_df.join([obs])
colors = ["k", "red", 'green']
plt.figure(figsize=(30, 15))
for i, column in enumerate(data.columns):
    plt.plot(
        data.index.values,
        data[column].values,
        linewidth=3,
        color=colors[i],
        label=column,
    )
plt.legend()
plt.ylabel(variable + r" ($\mu g / m^3$)", fontsize="xx-large")
plt.xlabel("Date", fontsize="x-large")
plt.title(
    f"{location_obj.city} ({location_obj.country})", fontsize="xx-large"
)
plt.show()
