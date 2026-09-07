def test_health_is_liveness_only(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_fails_when_elasticsearch_is_down(client_with_unready_repo):
    response = client_with_unready_repo.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "search_infrastructure_unavailable"


def test_ready_succeeds_when_repository_alias_and_embedding_are_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
