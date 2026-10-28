import pytest
from fastapi.testclient import TestClient

# Same example as the README; the real model is deterministic, so the output is stable.
VALID_PAYLOAD = {
    "target_datetime": "2018-07-15T14:00:00",
    "lag_24h": 42000,
    "lag_168h": 40500,
    "rolling_mean_24h": 35000,
}
README_EXAMPLE_MW = 40416.97

REQUIRED_FIELDS = list(VALID_PAYLOAD)
POSITIVE_FIELDS = ["lag_24h", "lag_168h", "rolling_mean_24h"]


def test_predict_valid_input_returns_forecast(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_mw"}
    assert isinstance(body["predicted_mw"], float)
    # Loose sanity bounds for PJM-East demand, to catch NaN, zero or garbage output.
    assert 10_000 < body["predicted_mw"] < 70_000


def test_predict_matches_readme_example(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.json()["predicted_mw"] == pytest.approx(README_EXAMPLE_MW, abs=0.01)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_predict_missing_field_returns_422(client: TestClient, field: str) -> None:
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != field}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("target_datetime", "not-a-date"),
        ("lag_24h", "abc"),
        ("lag_168h", None),
        ("rolling_mean_24h", [1, 2, 3]),
    ],
)
def test_predict_wrong_type_returns_422(client: TestClient, field: str, value: object) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, field: value})

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]


@pytest.mark.parametrize("field", POSITIVE_FIELDS)
@pytest.mark.parametrize("value", [0, -1, -42000])
def test_predict_non_positive_value_returns_422(client: TestClient, field: str, value: int) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, field: value})

    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]
