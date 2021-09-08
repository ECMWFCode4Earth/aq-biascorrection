import glob
import tempfile

import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner
from mockito import ANY, unstub, when

from src.constants import ROOT_DIR
from src.scripts import extraction_observations


def test_cli_download_openaq():
    data_stations = pd.read_csv(
        ROOT_DIR / "tests" / "data_test" / "stations.csv",
    )
    when(pd).read_csv(
        ANY(),
    ).thenReturn(data_stations)
    tempdir = tempfile.mkdtemp()
    runner = CliRunner()
    result = runner.invoke(extraction_observations.main, ["o3", "-o", tempdir])
    assert result.exit_code == 0

    # Check observations are converted to ug/m^3
    for file in glob.glob(tempdir + "/**/*.nc", recursive=True):
        ds = xr.open_dataset(file)
        assert ds.o3.attrs["units"] == "microgram / m^3"
    unstub()


# @pytest.skip("Prepare test ...")
# def test_cli_download_cams(mocker: MockerFixture):
#     runner = CliRunner()
#     result = runner.invoke(extraction_cams.main, [])
#     assert result.exit_code == 0
