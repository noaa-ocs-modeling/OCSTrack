
import pytest
from unittest.mock import patch, mock_open, MagicMock
from ocstrack.Observation.get_sat import download_sat_data
import requests

@pytest.fixture
def mock_requests_get():
    """Fixture to mock requests.get."""
    with patch('requests.get') as mock_get:
        yield mock_get

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_success(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test successful download of a single file.
    """
    # Configure the mock for a successful response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake', b'file', b'content']
    mock_requests_get.return_value.__enter__.return_value = mock_response

    # Use mock_open to simulate file writing
    m_open = mock_open()
    with patch('builtins.open', m_open):
        result = download_sat_data(
            dates_str=['20230101'],
            url_template='http://fake.url/',
            raw_dir='/fake/dir',
            sat='test_sat'
        )

        # Assertions
        mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
        mock_requests_get.assert_called_once_with('http://fake.url/20230101.nc', stream=True)
        m_open.assert_called_once_with('/fake/dir/20230101.nc', 'wb')
        handle = m_open()
        handle.write.assert_any_call(b'fake')
        handle.write.assert_any_call(b'content')
        assert result == ['/fake/dir/20230101.nc']

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_failure_and_retry(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test that the function retries on a request failure.
    """
    # Configure the mock to raise an exception
    mock_requests_get.side_effect = requests.exceptions.RequestException("Test Error")

    m_open = mock_open()
    with patch('builtins.open', m_open):
        with patch('time.sleep', return_value=None) as mock_sleep: # Mock time.sleep
            result = download_sat_data(
                dates_str=['20230101'],
                url_template='http://fake.url/',
                raw_dir='/fake/dir',
                sat='test_sat',
                retries=3,
                delay=1
            )

            # Assertions
            mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
            assert mock_requests_get.call_count == 3
            assert mock_sleep.call_count == 3 # Sleeps after each of the 3 failures
            m_open.assert_not_called() # File should not be opened on failure
            assert result == ['/fake/dir/20230101.nc'] # Still returns the expected path
