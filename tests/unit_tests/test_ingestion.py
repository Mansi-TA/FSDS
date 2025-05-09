import os
from unittest import mock

import pytest

from FSDS_.ingest import fetch_housing_data, load_housing_data

# Writing test for checking fetching housing data.


def test_fetch_housing_data():
    with mock.patch("os.makedirs") as mock_makedirs:

        with mock.patch("urllib.request.urlretrieve") as mock_urlretrieve:

            with mock.patch("tarfile.open") as mock_tarfile:

                fetch_housing_data()

                mock_makedirs.assert_called_once_with("datasets/housing", exist_ok=True)

                # Assert that urlretrieve was called with the right URL and path
                mock_urlretrieve.assert_called_once_with(
                    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
                    os.path.join("datasets", "housing", "housing.tgz"),
                )

                mock_tarfile.assert_called_once()
