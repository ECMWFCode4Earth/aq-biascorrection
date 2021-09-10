import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
from mockito import ANY, unstub, when

from src.data.utils import Location
from src.constants import ROOT_DIR
from src.metrics.validation_metrics import ValidationTables
from src.models.validation import ValidationDataset, Validator
from src.visualization.validation_visualization import ValidationVisualization


class TestValidation:
    @pytest.fixture()
    def mocked_validation_obj(self, tmp_path):
        tempdir = tmp_path / "sub"
        tempdir.mkdir()
        validation_obj = Validator(
            "InceptionTime_ensemble",
            "pm25",
            tempdir,
            tempdir,
        )
        return validation_obj

    @pytest.fixture()
    def mocked_visualization_obj(
        self, mocked_validation_obj, mocked_validation_datasets
    ):
        visual_obj = ValidationVisualization(
            mocked_validation_datasets,
            mocked_validation_obj.varname,
            Location.get_location_by_id("ES002"),
            "all",
            mocked_validation_obj.visualizations_output_dir,
        )
        return visual_obj

    @pytest.fixture()
    def mocked_tables_obj(self, mocked_validation_obj, mocked_validation_datasets):
        tables_obj = ValidationTables(
            mocked_validation_datasets,
            Location.get_location_by_id("ES002"),
            mocked_validation_obj.metrics_output_dir,
        )
        return tables_obj

    @pytest.fixture()
    def mocked_validation_datasets(self):
        nprandom = np.random.RandomState(42)
        validation_datasets_list = []
        for i in range(500):
            if i % 3 == 0:
                class_on_train = "test"
            else:
                class_on_train = "train"
            cams = nprandom.randint(70, 100, 24)
            cams_corrected = nprandom.randint(50, 65, 24)
            observations = nprandom.randint(45, 60, 24)
            persistence = nprandom.randint(45, 60, 24)
            start_time = datetime.datetime(
                year=2019, month=6, day=1
            ) + datetime.timedelta(days=i)
            end_time = start_time + datetime.timedelta(hours=23)
            times = pd.date_range(start_time, end_time, freq="H")
            cams_df = pd.DataFrame({"CAMS": cams}, index=times)
            cams_corrected_df = pd.DataFrame(
                {"Corrected CAMS": cams_corrected}, index=times
            )
            observations_df = pd.DataFrame({"Observations": observations}, index=times)
            persistence_df = pd.DataFrame({"Persistence": persistence}, index=times)
            val_ds = ValidationDataset(
                cams=cams_df,
                predictions=cams_corrected_df,
                observations=observations_df,
                persistence=persistence_df,
                class_on_train=class_on_train,
            )
            validation_datasets_list.append(val_ds)
        return validation_datasets_list

    def test_validation_workflow(
        self, mocked_validation_obj, mocked_validation_datasets
    ):
        station_id = "ES002"
        stations = pd.read_csv(
            ROOT_DIR / "tests" / "data_test" / "stations.csv",
            index_col=0,
            usecols=list(range(1, 8)),
        )
        station = stations[stations["location_id"] == station_id]
        dict_to_location = station.iloc[0].to_dict()
        loc_obj = Location(**dict_to_location)
        when(Location).get_location_by_id(ANY()).thenReturn(loc_obj)
        when(Validator).load_model_predictions(ANY(), ANY()).thenReturn(None)
        when(Validator).load_obs_and_cams(ANY()).thenReturn(None)
        when(Validator).get_initialization_datasets(ANY(), ANY()).thenReturn(
            mocked_validation_datasets
        )
        when(ValidationVisualization).run().thenReturn(None)
        when(ValidationTables).run().thenReturn(None)
        result = mocked_validation_obj.run(station_id)
        assert result is None
        unstub()

    def test_validation_tables_load_metrics_train_test_for_every_run(
        self, mocked_tables_obj
    ):
        train, test = mocked_tables_obj.load_metrics_train_test_for_every_run()
        assert type(train) is list
        assert type(test) is list
        assert type(train[0]) is pd.DataFrame
        assert type(test[0]) is pd.DataFrame
        train_data = pd.concat(train)
        test_data = pd.concat(test)
        assert type(train_data) is pd.DataFrame
        assert type(test_data) is pd.DataFrame
        assert train_data.shape == (999, 5)
        assert test_data.shape == (501, 5)
        assert list(train_data.columns) == [
            "NMAE",
            "BIAS",
            "RMSE",
            "De-Biased NMAE",
            "Pearson Correlation",
        ]
        assert list(test_data.columns) == [
            "NMAE",
            "BIAS",
            "RMSE",
            "De-Biased NMAE",
            "Pearson Correlation",
        ]

    def test_validation_tables_load_data_train_test_for_entire_set(
        self, mocked_tables_obj
    ):
        train, test = mocked_tables_obj.load_data_train_test_for_entire_set()
        assert type(train) is list
        assert type(test) is list
        assert type(train[0]) is pd.DataFrame
        assert type(test[0]) is pd.DataFrame
        train_data = pd.concat(train)
        test_data = pd.concat(test)
        assert type(train_data) is pd.DataFrame
        assert type(test_data) is pd.DataFrame
        assert train_data.shape == (7992, 4)
        assert test_data.shape == (4008, 4)
        assert list(train_data.columns) == [
            "CAMS",
            "Observations",
            "Corrected CAMS",
            "Persistence",
        ]
        assert list(test_data.columns) == [
            "CAMS",
            "Observations",
            "Corrected CAMS",
            "Persistence",
        ]

    def test_validation_tables_load_metrics_train_test_for_entire_set_aggregated(
        self, mocked_tables_obj
    ):
        train, test = mocked_tables_obj.load_data_train_test_for_entire_set()
        train_data = pd.concat(train)
        test_data = pd.concat(test)
        dict_agg = mocked_tables_obj.load_metrics_train_test_for_entire_set_aggregated(
            train_data, test_data
        )
        assert list(dict_agg.keys()) == ["train", "test"]
        assert type(dict_agg["train"]) is list
        assert type(dict_agg["test"]) is list
        assert type(dict_agg["train"][0]) is pd.DataFrame
        assert type(dict_agg["test"][0]) is pd.DataFrame
        assert list(dict_agg["train"][0]) == [
            "NMAE",
            "BIAS",
            "RMSE",
            "De-Biased NMAE",
            "Pearson Correlation",
        ]
        assert list(dict_agg["test"][0]) == [
            "NMAE",
            "BIAS",
            "RMSE",
            "De-Biased NMAE",
            "Pearson Correlation",
        ]

    def test_validation_tables_run_for_every_prediction(self, mocked_tables_obj):
        output_paths = mocked_tables_obj.run_for_every_prediction()
        assert len(output_paths)
        for output_path in output_paths:
            assert output_path.exists()

    def test_validation_tables_run_for_the_complete_data(self, mocked_tables_obj):
        output_paths = mocked_tables_obj.run_for_the_complete_data()
        assert len(output_paths)
        for output_path in output_paths:
            assert output_path.exists()

    def test_validation_visualizations_get_dataset_for_timeseries(
        self, mocked_visualization_obj
    ):
        data = mocked_visualization_obj.get_dataset_for_timeseries(
            mocked_visualization_obj.validation_datasets
        )
        assert list(data.columns) == [
            "CAMS",
            "Observations",
            "Corrected CAMS",
            "Persistence",
        ]
        assert data.shape == (12000, 4)
        assert type(data) is pd.DataFrame

    def test_validation_visualization_get_dataset_for_boxplot(
        self, mocked_visualization_obj
    ):
        data = mocked_visualization_obj.get_dataset_for_boxplot(
            mocked_visualization_obj.validation_datasets
        )
        assert list(data.columns) == [
            "Class on train",
            "Data Type",
            "pm25 ($\\mu g / m^3$)",
            "Data Kind",
        ]
        assert data.shape == (48000, 4)
        assert type(data) is pd.DataFrame
        assert list(np.unique(data["Class on train"].values)) == ["test", "train"]
        assert list(np.unique(data["Data Kind"].values)) == [
            "CAMS - (test)",
            "CAMS - (train)",
            "Corrected CAMS - (test)",
            "Corrected CAMS - (train)",
            "Observations - (test)",
            "Observations - (train)",
            "Persistence - (test)",
            "Persistence - (train)",
        ]

    def test_validation_visualization_run(self, mocked_visualization_obj):
        output_paths = mocked_visualization_obj.run()
        assert len(output_paths)
        for output_path in output_paths:
            assert output_path.exists()
