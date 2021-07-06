import filecmp
import glob
import tempfile

import pytest
import pandas as pd
import xarray as xr

from src.scripts import extraction_openaq, extraction_cams
from tests.file_provider import get_remote_file

from pytest_mock import MockerFixture
from click.testing import CliRunner


def test_cli_download_openaq(mocker: MockerFixture):
    mocker.patch.object(
        extraction_openaq.pd, 
        'read_csv',
        return_value=pd.DataFrame({
            'id': ['AT001', 'AU005'], 'city': ['Vienna', 'Melbourne'], 
            'country': ['Austria', 'Australia'], 'elevation': [189, 27], 
            'longitude': [16.37208, 144.96332], 'latitude': [48.20849, -37.814], 
            'timezone': ['Europe/Vienna', 'Australia/Melbourne']
        })
    )
    tempdir = tempfile.mkdtemp()
    runner = CliRunner()
    result = runner.invoke(extraction_openaq.main, ['o3', '-o', tempdir])
    assert result.exit_code == 0

    # Check observations are converted to ug/m^3
    for file in glob.glob(tempdir + "/**/*.nc", recursive=True):
        ds = xr.open_dataset(file)
        assert ds.o3.attrs['units'] == 'microgram / m^3'


@pytest.skip("Prepare test ...")
def test_cli_download_cams(mocker: MockerFixture):
    runner = CliRunner()
    result = runner.invoke(extraction_cams.main, [])
    assert result.exit_code == 0
