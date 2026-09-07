from __future__ import annotations
import json
from shopmind.app.llm.base import InvalidLLMOutput, LLMProviderError, LLMRequest, LLMResponse
class OpenAIProvider:
    def __init__(self, *, client=None, model="gpt-5-mini", api_key=None):
        self.model=model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e: raise LLMProviderError("openai package is required") from e
            client=OpenAI(api_key=api_key)
        self.client=client
    @property
    def is_ready(self): return self.client is not None and bool(self.model)
    def generate(self,request:LLMRequest)->LLMResponse:
        schema={"type":"object","properties":{"answer":{"type":"string"},"citation_ids":{"type":"array","items":{"type":"string"}},"insufficient_evidence":{"type":"boolean"}},"required":["answer","citation_ids","insufficient_evidence"],"additionalProperties":False}
        try:
            response=self.client.responses.create(model=self.model,store=False,input=[{"role":"system","content":request.instructions},{"role":"user","content":f"Question: {request.question}\n\nEvidence:\n{request.rendered_context}\n\nAllowed citation IDs: {list(request.allowed_citation_ids)}"}],text={"format":{"type":"json_schema","name":request.schema_version,"schema":schema,"strict":True}})
        except Exception as e: raise LLMProviderError("OpenAI Responses request failed") from e
        try:
            raw=json.loads(response.output_text)
            return LLMResponse(str(raw["answer"]),tuple(str(x) for x in raw["citation_ids"]),bool(raw["insufficient_evidence"]),{"provider":"openai","model":getattr(response,"model",self.model),"response_id":getattr(response,"id",None)})
        except Exception as e: raise InvalidLLMOutput("OpenAI structured response is invalid") from e
