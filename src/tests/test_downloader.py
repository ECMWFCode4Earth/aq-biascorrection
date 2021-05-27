from src.data.download.downloader import OpenAQDownloader
from src.data.utils import Location
from src.tests.file_provider import get_remote_file

import tempfile
import filecmp

tempdir = tempfile.mkdtemp()


def test_download_aq():
    latitude_dubai = 25.0657
    longitude_dubai = 55.17128
    filename = "dubai/ae001/o3_dubai_ae001_20190101_20210331"
    data_filename = filename + ".csv"
    metadata_filename = filename + "_metadata.csv"
    
    # Load the data into the temporary directory.        
    get_remote_file(data_filename, tempdir)
    get_remote_file(metadata_filename, tempdir)

    loc = Location(
        "AE001", "Dubai", "United Arab Emirates", 
        latitude_dubai, longitude_dubai)
    OpenAQDownloader(loc, tempdir + "/downloaded", 'o3').run() 
    
    assert filecmp.dircmp(tempdir + "/downloaded", tempdir)
    assert filecmp.cmp(f"{tempdir}/downloaded/{data_filename}", 
                            f"{tempdir}/{data_filename}")
    assert filecmp.cmp(f"{tempdir}/downloaded/{metadata_filename}", 
                            f"{tempdir}/{metadata_filename}")
