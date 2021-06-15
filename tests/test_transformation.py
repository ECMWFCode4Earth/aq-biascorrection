from pytest_mock import MockerFixture

from src.data.transformation.trandormation_location_data import LocationTransformer
from src.data.utils import Location


def test_location_transformation(mocker: MockerFixture):
    loc = Location(
        'TEST01',
        'Madrid',
        'Spain',
        40.4165,
        -3.70256,
        'Europe/Madrid',
        668)
    result = LocationTransformer('pm25', loc).run()