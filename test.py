import pytest
import json
from app import app  # Ensure 'app.py' contains `app = Flask(__name__)`

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Random Forest API is running' in response.data  # updated message check

def test_predict_valid(client):
    sequence = list(range(7))
    response = client.post(
        '/predict',
        data=json.dumps({'sequence': sequence}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], list)
    assert len(data['prediction']) == 1  # Because model outputs 13 steps

def test_predict_missing_sequence(client):
    response = client.post(
        '/predict',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'Missing "sequence"' in data['error']

def test_predict_wrong_length(client):
    sequence = list(range(10))  # Incorrect length
    response = client.post(
        '/predict',
        data=json.dumps({'sequence': sequence}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'Sequence must be of length' in data['error']
