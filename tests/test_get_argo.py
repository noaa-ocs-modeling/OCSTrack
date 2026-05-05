
from ocstrack.Observation.get_argo import generate_monthly_dates

def test_generate_monthly_dates():
    """
    Test the generate_monthly_dates function for a multi-month range.
    """
    start_date = "2023-01-15"
    end_date = "2023-03-20"
    
    expected_months = [
        ("2023", "01"),
        ("2023", "02"),
        ("2023", "03"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_single_month():
    """
    Test the function for a date range within a single month.
    """
    start_date = "2023-01-10"
    end_date = "2023-01-20"
    
    expected_months = [("2023", "01")]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_year_boundary():
    """
    Test the function across a year boundary.
    """
    start_date = "2022-12-01"
    end_date = "2023-02-01"
    
    expected_months = [
        ("2022", "12"),
        ("2023", "01"),
        ("2023", "02"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months


from ocstrack.Observation.get_argo import _download_file
from unittest.mock import patch, mock_open, MagicMock
import requests

@patch('requests.get')
def test_download_file_success(mock_get):
    """
    Test the _download_file helper for a successful download.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake data']
    mock_get.return_value.__enter__.return_value = mock_response

    m_open = mock_open()
    with patch('builtins.open', m_open):
        success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

        assert success is True
        mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True)
        m_open.assert_called_once_with('/fake/path/file.nc', 'wb')
        m_open().write.assert_called_once_with(b'fake data')

@patch('requests.get')
@patch('os.path.exists', return_value=True)
@patch('os.remove')
def test_download_file_failure(mock_remove, mock_exists, mock_get):
    """
    Test the _download_file helper for a failed download.
    It should return False and clean up partially downloaded files.
    """
    mock_get.side_effect = requests.exceptions.RequestException('Test Error')

    # The function should catch the exception and return False
    success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

    assert success is False
    mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True)
    # Check that it attempts to clean up the file
    mock_remove.assert_called_once_with('/fake/path/file.nc')
