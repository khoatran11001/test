from shopmind.app.rag.models import Citation
class CitationValidationError(ValueError): pass
class CitationValidator:
    def validate(self,citation_ids,allowed_ids,*,answer:str,insufficient_evidence:bool):
        unknown=[x for x in citation_ids if x not in allowed_ids]
        if unknown: raise CitationValidationError(f"unknown citation: {unknown[0]}")
        deduped=tuple(dict.fromkeys(citation_ids))
        if not insufficient_evidence and answer.strip() and not deduped: raise CitationValidationError("answer requires at least one citation")
        return deduped
class CitationResolver:
    def resolve(self,citation_ids,documents):
        by_id={d.id:d for d in documents}; out=[]
        for cid in citation_ids:
            d=by_id[cid]; m=d.metadata
            out.append(Citation(d.id,d.source,str(m.get("title") or d.id),m.get("product_id"),m.get("version"),m.get("source_url"),{k:m[k] for k in sorted(m) if k in {"rating","verified_purchase","helpful_vote","matched_by","effective_date","policy_type"}}))
        return tuple(out)
