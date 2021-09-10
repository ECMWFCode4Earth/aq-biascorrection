from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from src import constants
from src.scripts.plotting import main_corrs, main_hourly_bias, main_line
from src.visualization import data_visualization

@pytest.fixture()
def mock_data():
    nprandom = np.random.RandomState(42)
    times = pd.date_range("2020-06-01", "2021-03-31", freq="1H")
    dicts_to_df = []
    for i in range(len(times)):
        dicts_to_df.append(
            {
                "index": times[i],
                "blh_forecast": 166.61845 * nprandom.uniform(0.5, 1.5),
                "d2m_forecast": 277.01556 * nprandom.uniform(0.5, 1.5),
                "dsrp_forecast": 1562624.0 * nprandom.uniform(0.5, 1.5),
                "o3_forecast": 74.84656 * nprandom.uniform(0.5, 1.5),
                "msl_forecast": 101776.5 * nprandom.uniform(0.5, 1.5),
                "no2_forecast": 5.846614 * nprandom.uniform(0.5, 1.5),
                "pm10_forecast": 11.262489 * nprandom.uniform(0.5, 1.5),
                "pm25_forecast": 7.691787 * nprandom.uniform(0.5, 1.5),
                "so2_forecast": 1.0124128 * nprandom.uniform(0.5, 1.5),
                "t2m_forecast": 280.02136 * nprandom.uniform(0.5, 1.5),
                "tcc_forecast": 0.99993896 * nprandom.uniform(0.1, 0.9),
                "tp_forecast": 0.0007171631 * nprandom.uniform(0.5, 1.5),
                "u10_forecast": -2.9277039 * nprandom.uniform(0.5, 1.5),
                "uvb_forecast": 549440.0 * nprandom.uniform(0.5, 1.5),
                "v10_forecast": -0.1618042 * nprandom.uniform(0.5, 1.5),
                "z_forecast": 54970.42 * nprandom.uniform(0.5, 1.5),
                "pm25_observed": 5.120075757575758 * nprandom.uniform(0.5, 1.5),
                "local_time_hour": times[i].hour,
                "pm25_bias": 2.5717110084764885 * nprandom.uniform(0.5, 1.5),
            }
        )
    return pd.DataFrame(dicts_to_df)

@pytest.fixture()
def mock_metadata():
    metadata_dict = {
        "id": {28: "CA001", 29: "CA002", 30: "CA003"},
        "city": {28: "Montreal", 29: "Toronto", 30: "Vancouver"},
        "country": {28: "Canada", 29: "Canada", 30: "Canada"},
        "latitude": {28: 45.50884, 29: 43.70011, 30: 49.24966},
        "longitude": {28: -73.58781, 29: -79.4163, 30: -123.11934},
        "timezone": {
            28: "America/Toronto",
            29: "America/Toronto",
            30: "America/Vancouver",
        },
        "elevation": {28: 217, 29: 170, 30: 76},
    }
    return pd.DataFrame(metadata_dict)


def test_line_plot(mocker: MockerFixture, mock_data, mock_metadata):
    mocker.patch.object(
        data_visualization.pd, "read_csv", side_effect=[mock_metadata, mock_data]
    )
    data_visualization.StationTemporalSeriesPlotter(
        "pm25",
        "Canada",
        Path(constants.ROOT_DIR) / "data" / "processed",
        stations=["Montreal"],
    ).plot_data()
    data_visualization.pd.read_csv.assert_called()


def test_heatmap_corrs(mocker: MockerFixture, mock_data, mock_metadata):
    mocker.patch.object(
        data_visualization.pd, "read_csv", side_effect=[mock_metadata, mock_data]
    )
    data_visualization.StationTemporalSeriesPlotter(
        "pm25",
        "Canada",
        Path(constants.ROOT_DIR) / "data" / "processed",
        stations=["Montreal"],
    ).plot_correlations()
    data_visualization.pd.read_csv.assert_called()


def test_hourly_bias_plot(mocker: MockerFixture, mock_data, mock_metadata):
    mocker.patch.object(
        data_visualization.pd,
        "read_csv",
        side_effect=[mock_metadata, mock_data, mock_data, mock_data],
    )
    data_visualization.StationTemporalSeriesPlotter(
        "pm25", "Canada", Path(constants.ROOT_DIR) / "data" / "processed"
    ).plot_hourly_bias()
    data_visualization.pd.read_csv.assert_called()


# def test_cli_line_plot(mocker: MockerFixture, mock_data, mock_metadata):
#     mocker.patch.object(
#         data_visualization.pd, "read_csv", side_effect=[mock_metadata, mock_data]
#     )
#     runner = CliRunner()
#     result = runner.invoke(
#         main_line,
#         [
#             "pm25",
#             "Canada",
#             "-d",
#             str(Path(constants.ROOT_DIR) / "data" / "processed"),
#             "-s",
#             "Montreal",
#         ],
#     )
#
#     assert result.exit_code == 0
#
#
# def test_cli_plot_heatmap(mocker: MockerFixture, mock_data, mock_metadata):
#     mocker.patch.object(
#         data_visualization.pd, "read_csv", side_effect=[mock_metadata, mock_data]
#     )
#     runner = CliRunner()
#     result = runner.invoke(
#         main_corrs,
#         [
#             "pm25",
#             "Canada",
#             "-d",
#             str(Path(constants.ROOT_DIR) / "data" / "processed"),
#             "-s",
#             "Montreal",
#         ],
#     )
#
#     assert result.exit_code == 0
#
#
# def test_cli_hourly_bias_plot(mocker: MockerFixture, mock_data, mock_metadata):
#     mocker.patch.object(
#         data_visualization.pd,
#         "read_csv",
#         side_effect=[mock_metadata, mock_data, mock_data, mock_data],
#     )
#     runner = CliRunner()
#     result = runner.invoke(
#         main_hourly_bias,
#         ["pm25", "Canada", "-d", str(Path(constants.ROOT_DIR) / "data" / "processed")],
#     )
#
#     assert result.exit_code == 0
