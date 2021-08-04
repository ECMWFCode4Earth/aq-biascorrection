from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from src import constants
from src.scripts.plotting import main_corrs, main_hourly_bias, main_line
from src.visualization import data_visualizer


def mock_data():
    return pd.read_csv("tests/data/ploting_fake.csv", index_col=0)


def mock_metadata():
    return pd.read_csv("tests/data/metadata_canada.csv", index_col=0)


def test_line_plot(mocker: MockerFixture):
    mocker.patch.object(
        data_visualizer.pd, "read_csv", side_effect=[mock_metadata(), mock_data()]
    )
    data_visualizer.StationTemporalSeriesPlotter(
        "pm25",
        "Canada",
        Path(constants.ROOT_DIR) / "data" / "processed",
        stations=["Montreal"],
    ).plot_data()
    data_visualizer.pd.read_csv.assert_called()


def test_heatmap_corrs(mocker: MockerFixture):
    mocker.patch.object(
        data_visualizer.pd, "read_csv", side_effect=[mock_metadata(), mock_data()]
    )
    data_visualizer.StationTemporalSeriesPlotter(
        "pm25",
        "Canada",
        Path(constants.ROOT_DIR) / "data" / "processed",
        stations=["Montreal"],
    ).plot_correlations()
    data_visualizer.pd.read_csv.assert_called()


def test_hourly_bias_plot(mocker: MockerFixture):
    mocker.patch.object(
        data_visualizer.pd,
        "read_csv",
        side_effect=[mock_metadata(), mock_data(), mock_data(), mock_data()],
    )
    data_visualizer.StationTemporalSeriesPlotter(
        "pm25", "Canada", Path(constants.ROOT_DIR) / "data" / "processed"
    ).plot_hourly_bias()
    data_visualizer.pd.read_csv.assert_called()


def test_cli_line_plot(mocker: MockerFixture):
    mocker.patch.object(
        data_visualizer.pd, "read_csv", side_effect=[mock_metadata(), mock_data()]
    )
    runner = CliRunner()
    result = runner.invoke(
        main_line,
        [
            "pm25",
            "Canada",
            "-d",
            str(Path(constants.ROOT_DIR) / "data" / "processed"),
            "-s",
            "Montreal",
        ],
    )

    assert result.exit_code == 0


def test_cli_plot_heatmap():
    runner = CliRunner()
    result = runner.invoke(
        main_corrs,
        [
            "pm25",
            "Canada",
            "-d",
            str(Path(constants.ROOT_DIR) / "data" / "processed"),
            "-s",
            "Montreal",
        ],
    )

    assert result.exit_code == 0


def test_cli_hourly_bias_plot(mocker: MockerFixture):
    mocker.patch.object(
        data_visualizer.pd,
        "read_csv",
        side_effect=[mock_metadata(), mock_data(), mock_data(), mock_data()],
    )
    runner = CliRunner()
    result = runner.invoke(
        main_hourly_bias,
        ["pm25", "Canada", "-d", str(Path(constants.ROOT_DIR) / "data" / "processed")],
    )

    assert result.exit_code == 0
