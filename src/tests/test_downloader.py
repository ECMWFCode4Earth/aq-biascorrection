from src.data.downloader import Location, OpenAQDownloader
from src.tests.file_provider import get_remote_file
from pathlib import Path

import tempfile


class TestDownloaderAQ:
    def test_download_aq():
        tempdir = tempfile.mkdtemp()
        get_remote_file("dubai", tempdir)

        latitude_dubai = 25.0657
        longitude_dubai = 55.17128
        data_path = Path('/tmp/test_download_aq/')
        
        loc = Location(
            "AE001", "Dubai", "United Arab Emirates", 
            latitude_dubai, longitude_dubai)
        OpenAQDownloader(loc, data_path, 'o3').run() 


if __name__ == '__main__':
    TestDownloaderAQ().test_download_aq()
