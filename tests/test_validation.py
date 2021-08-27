import datetime

import numpy as np
import pandas as pd
import pytest
import pathlib

from mockito import ANY, mock, when

from src.constants import ROOT_DIR
from src.data.utils import Location
from src.models.validation import Validator, ValidationDataset
from src.metrics.validation_metrics import ValidationTables
from src.visualization.validation_visualization import ValidationVisualization


class TestValidation:
    @pytest.fixture()
    def mocked_validation_obj(self):
        validation_obj = Validator(
            "InceptionTime_ensemble",
            "pm25",
            ROOT_DIR / "reports" / "figures" / "results",
            ROOT_DIR / "reports" / "tables" / "results",
        )
        return validation_obj

    @pytest.fixture()
    def mocked_visualization_obj(
            self,
            mocked_validation_obj,
            mocked_validation_datasets
    ):
        visual_obj = ValidationVisualization(
            mocked_validation_datasets,
            mocked_validation_obj.varname,
            Location.get_location_by_id('ES002'),
            "all",
            mocked_validation_obj.visualizations_output_dir
        )
        return visual_obj

    @pytest.fixture()
    def mocked_tables_obj(
            self,
            mocked_validation_obj,
            mocked_validation_datasets
    ):
        tables_obj = ValidationTables(
            mocked_validation_datasets,
            Location.get_location_by_id('ES002'),
            mocked_validation_obj.metrics_output_dir
        )
        return tables_obj

    @pytest.fixture()
    def mocked_validation_datasets(self):
        validation_datasets_list = []
        for i in range(500):
            cams = np.random.randint(70, 100, 24)
            cams_corrected = np.random.randint(50, 65, 24)
            observations = np.random.randint(45, 60, 24)
            persistence = np.random.randint(45, 60, 24)
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
                class_on_train=np.random.choice(np.array(['train', 'train', 'test'])),
            )
            validation_datasets_list.append(val_ds)
        return validation_datasets_list

    def test_validation_workflow(
            self,
            mocked_validation_obj,
            mocked_tables_obj,
            mocked_visualization_obj,
            mocked_validation_datasets
    ):
        station_id = "ES002"
        when(mocked_validation_obj).load_model_predictions(ANY(), ANY()).thenReturn(
            None
        )
        when(mocked_validation_obj).load_obs_and_cams(ANY()).thenReturn(None)
        when(mocked_validation_obj).get_initialization_datasets(
            ANY(), ANY()
        ).thenReturn(mocked_validation_datasets)
        when(mocked_tables_obj).run().thenReturn(None)
        when(mocked_visualization_obj).run().thenReturn(None)
        mocked_validation_obj.run(station_id)
