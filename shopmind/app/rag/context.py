from __future__ import annotations
from shopmind.app.rag.models import RAGContext
class ContextBuilder:
    def __init__(self,max_documents:int,source_limits:dict[str,int],max_characters:int):
        if max_documents<=0 or max_characters<=0: raise ValueError('context limits must be positive')
        if any(v<0 for v in source_limits.values()): raise ValueError('source limits must be non-negative')
        self.max_documents=max_documents;self.source_limits=dict(source_limits);self.max_characters=max_characters
    def build(self,*args):
        if len(args)==1: question='';documents=args[0]
        elif len(args)==2: question,documents=args
        else: raise TypeError('build expects documents or question, documents')
        counts={};chosen=[];chunks=[];used=0
        for doc in documents:
            limit=self.source_limits.get(doc.source,self.max_documents)
            if counts.get(doc.source,0)>=limit:continue
            title=str(doc.metadata.get('title') or doc.id)
            chunk=f'[{doc.id}]\nTitle: {title}\nSource: {doc.source}\n{doc.content.strip()}'
            extra=len(chunk)+(2 if chunks else 0)
            if used+extra>self.max_characters:continue
            chunks.append(chunk);chosen.append(doc);counts[doc.source]=counts.get(doc.source,0)+1;used+=extra
            if len(chosen)>=self.max_documents:break
        return RAGContext(tuple(chosen),'\n\n'.join(chunks),tuple(d.id for d in chosen),{'source_counts':counts,'context_characters':used,'question':str(question)})
