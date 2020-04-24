"""Tests for '/recommend' endpoint in allay-ds-api/fastapi_app.

To run:
- Enter virtual environment
- change to `allay-ds-api` directory
- `python -m pytest ../tests/test_recommend.py`
"""

from json import loads

from fastapi.testclient import TestClient

from fastapi_app import APP

client = TestClient(APP)
endpoint_path = '/recommend'


def test_post():
    """Test a valid POST request to the /recommend endpoint."""
    test_data = {
        'user_id': 1,
        'post_ids': [1, 2, 3]
    }
    # pass a dict in the 'json' parameter to pass a JSON body
    response = client.post(endpoint_path, json=test_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['user_id'] == test_data['user_id']
    assert response_data['post_ids'] == test_data['post_ids']


def test_get():
    """Test a GET request to the /recommend endpoint - invalid."""
    response = client.get(endpoint_path)
    # expect HTTP response code 405 - Method Not Allowed
    assert response.status_code == 405

def test_post_no_user_id():
    """Test an invalid POST request to the /recommend endpoint."""
    test_data = {
        'error': "no 'user_id' key",
        'post_ids': [1, 2, 3]
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422

def test_post_no_post_ids():
    """Test an invalid POST request to the /recommend endpoint."""
    test_data = {
        'error': "no 'post_ids' key",
        'user_id': 1
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422

def test_post_user_id_not_int():
    """Test an invalid POST request to the /recommend endpoint."""
    test_data = {
        'user_id': 'not an integer'
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422

def test_post_post_ids_not_int_list():
    """Test an invalid POST request to the /recommend endpoint."""
    test_data = {
        'post_ids': ['not an list of integers', 42],
        'user_id': 1
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422
