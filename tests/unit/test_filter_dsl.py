import pytest
from shopmind.app.infrastructure.elasticsearch.repository import build_filter_clauses


def test_filters_support_single_and_multi_values():
    clauses = build_filter_clauses({"brand":"Acme","category":["Shoes","Boots"]})
    assert {"term":{"brand.keyword":"Acme"}} in clauses
    assert {"terms":{"category.keyword":["Shoes","Boots"]}} in clauses


def test_product_id_uses_keyword_field_directly():
    assert build_filter_clauses({"product_id":"P1"}) == [{"term":{"product_id":"P1"}}]


def test_arbitrary_filter_field_is_rejected():
    with pytest.raises(ValueError, match="unsupported filter field"): build_filter_clauses({"metadata.secret":"x"})
