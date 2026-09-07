from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from shopmind.app.main import create_app
from tests.api.conftest import FakeEmbedder, FakeRepository, FakeSearchService


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def test_text_search_returns_common_result_shape(client):
    response = client.post("/api/v1/search/text", json={"query": "black running shoes", "mode": "hybrid", "top_k": 10, "filters": {"category": "Shoes"}})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "black running shoes"
    assert body["mode"] == "hybrid"
    assert isinstance(body["results"], list)
    assert set(body["results"][0]) == {"product_id", "title", "image_url", "brand", "category", "score", "rank"}
    assert "retrieval_scores" not in body["results"][0]


def test_text_search_validation_statuses(client):
    assert client.post("/api/v1/search/text", json={"query": "   "}).status_code == 422
    assert client.post("/api/v1/search/text", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/search/text", json={"query": "x", "top_k": 10, "candidate_k": 5}).status_code == 422
    assert client.post("/api/v1/search/text", json={"query": "x", "mode": "not-a-mode"}).status_code == 400
    assert client.post("/api/v1/search/text", json={"query": "x", "filters": {"metadata.secret": "x"}}).status_code == 400


def test_empty_results_are_http_200():
    service = FakeSearchService(results=[])
    app = create_app(search_service=service, repository=FakeRepository(), embedder=FakeEmbedder(), auto_wire=False)
    with TestClient(app) as client:
        response = client.post("/api/v1/search/text", json={"query": "nothing"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_valid_image_upload_decodes_rgb_before_service(client, fake_service):
    response = client.post("/api/v1/search/image?top_k=3&candidate_k=8&category=Shoes", files={"image": ("query.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    decoded, top_k, candidate_k, filters = fake_service.image_requests[0]
    assert decoded.mode == "RGB"
    assert (top_k, candidate_k) == (3, 8)
    assert filters == {"category": "Shoes"}


def test_non_image_content_and_invalid_image_bytes_fail(client):
    wrong_type = client.post("/api/v1/search/image", files={"image": ("x.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 400
    invalid_bytes = client.post("/api/v1/search/image", files={"image": ("x.png", b"not an image", "image/png")})
    assert invalid_bytes.status_code == 400


def test_oversized_image_is_413():
    app = create_app(search_service=FakeSearchService(), repository=FakeRepository(), embedder=FakeEmbedder(), max_image_bytes=8, auto_wire=False)
    with TestClient(app) as client:
        response = client.post("/api/v1/search/image", files={"image": ("x.png", _png_bytes(), "image/png")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_request_id_is_returned_and_client_value_is_preserved(client):
    response = client.post("/api/v1/search/text", headers={"X-Request-ID": "req-test-123"}, json={"query": "runner"})
    assert response.headers["X-Request-ID"] == "req-test-123"
