"""Tests for '/check_review' endpoint in allay-ds-api/fastapi_app.

To run:
- Enter virtual environment
- change to `allay-ds-api` directory
- `python -m pytest ../tests/test_check_review.py`
"""

from json import loads

from fastapi.testclient import TestClient

from fastapi_app import APP

client = TestClient(APP)
endpoint_path = '/check_review'


def test_post():
    """Test a valid POST request to the /check_review endpoint."""
    test_data = {
        'comment': 'Everything was wonderful'
    }
    # pass a dict in the 'json' parameter to pass a JSON body
    response = client.post(endpoint_path, json=test_data)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data['comment'] == test_data['comment']
    assert isinstance(response_data['flag'], int)
    assert response_data['flag'] >= 0 and response_data['flag'] <= 2
    assert isinstance(response_data['score'], float)
    assert response_data['score'] >= 0.0 and response_data['score'] <= 1.0


def test_get():
    """Test a GET request to the /check_review endpoint - invalid."""
    response = client.get(endpoint_path)
    # expect HTTP response code 405 - Method Not Allowed
    assert response.status_code == 405

def test_post_no_comment():
    """Test an invalid POST request to the /check_review endpoint."""
    test_data = {
        'error': "no 'comment' key"
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422

def test_post_comment_not_string():
    """Test an invalid POST request to the /check_review endpoint."""
    test_data = {
        'comment': [42, 'joe', 'fred']
    }
    response = client.post(endpoint_path, json=test_data)
    # expect HTTP response code 422 - Unprocessable Entity
    assert response.status_code == 422
