ERROR_CATEGORIES={"source_retrieval_miss","wrong_source_dominance","product_review_mismatch","poor_dense_semantic_match","lexical_ambiguity","reranker_regression","context_truncation_or_diversity_loss","missing_required_policy_evidence","answer_unsupported_by_context","invalid_citation_id","correct_citation_but_unsupported_claim","answerable_marked_insufficient","unanswerable_answered_definitively","none"}
def categorize_error(*,expected_ids,retrieved_ids,context_ids,returned_ids,expected_insufficient,predicted_insufficient,answer_supported):
    if expected_ids and not set(expected_ids)&set(retrieved_ids): return "source_retrieval_miss"
    if expected_ids and not set(expected_ids)&set(context_ids): return "context_truncation_or_diversity_loss"
    if any(x.startswith("policy:") for x in expected_ids) and not any(x.startswith("policy:") for x in context_ids): return "missing_required_policy_evidence"
    if any(x not in context_ids for x in returned_ids): return "invalid_citation_id"
    if expected_insufficient and not predicted_insufficient: return "unanswerable_answered_definitively"
    if not expected_insufficient and predicted_insufficient: return "answerable_marked_insufficient"
    if not answer_supported: return "answer_unsupported_by_context"
    return "none"
