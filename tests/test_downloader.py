import glob
import tempfile

import xarray as xr
from click.testing import CliRunner

from src.constants import ROOT_DIR
from src.scripts import extraction_observations


def test_cli_download_openaq():
    tempdir = tempfile.mkdtemp()
    runner = CliRunner()
    result = runner.invoke(
        extraction_observations.main,
        ["o3", "-l", ROOT_DIR / "tests" / "data_test" / "stations.csv", "-o", tempdir]
    )
    assert result.exit_code == 0

    # Check observations are converted to ug/m^3
    for file in glob.glob(tempdir + "/**/*.nc", recursive=True):
        ds = xr.open_dataset(file)
        assert ds.o3.attrs["units"] == "microgram / m^3"