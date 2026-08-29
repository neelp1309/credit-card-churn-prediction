def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_success(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_flag"], bool)
    assert 0.0 <= body["threshold_used"] <= 1.0
    assert len(body["top_reasons"]) == 3


def test_threshold_validation(client, valid_payload):
    response = client.post("/predict?threshold=2", json=valid_payload)
    assert response.status_code == 422


def test_negative_threshold_validation(client, valid_payload):
    response = client.post("/predict?threshold=-0.01", json=valid_payload)
    assert response.status_code == 422


def test_rejects_unknown_fields(client, valid_payload):
    payload = {**valid_payload, "unexpected": "field"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_revolving_balance_cannot_exceed_limit(client, valid_payload):
    payload = {**valid_payload, "Credit_Limit": 1000.0, "Total_Revolving_Bal": 1200.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_api_requests_total" in response.text
