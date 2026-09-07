from __future__ import annotations
import json
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any
import yaml
from shopmind.pipeline.knowledge_document import KnowledgeDocument
@dataclass(frozen=True)
class PolicySection:
    policy_id:str;section_id:str;title:str;content:str;policy_type:str;version:str;effective_date:str;source_url:str|None;metadata:dict[str,Any]
    @property
    def citation_id(self):return f'policy:{self.section_id}'
    @property
    def search_text(self):return f'{self.title}\n{self.content}'.strip()
    def to_dict(self):return asdict(self)
def load_policy_sections(path):
    path=Path(path);rows=[];seen=set()
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        raw=json.loads(line);required=('policy_id','section_id','title','content','policy_type','version','effective_date');missing=[f for f in required if not str(raw.get(f) or '').strip()]
        if missing:raise ValueError(f'{path}:{n} missing required fields: {",".join(missing)}')
        sid=str(raw['section_id'])
        if sid in seen:raise ValueError(f'duplicate policy section_id: {sid}')
        seen.add(sid);rows.append(PolicySection(str(raw['policy_id']),sid,str(raw['title']).strip(),str(raw['content']).strip(),str(raw['policy_type']),str(raw['version']),str(raw['effective_date']),None if raw.get('source_url') is None else str(raw['source_url']),dict(raw.get('metadata') or {})))
    return rows
def _load_yaml_dir(root:Path):
    sections=[]
    for path in sorted(list(root.glob('*.yaml'))+list(root.glob('*.yml'))):
        raw=yaml.safe_load(path.read_text()) or {};pid=str(raw['policy_id']);version=str(raw['version']);effective=str(raw.get('effective_date') or '');url=raw.get('source_url')
        for s in raw.get('sections') or []:
            sid=str(s['section_id']); full=sid if '.' in sid else f'{pid}.{sid}';sections.append(PolicySection(pid,full,str(s['title']),str(s['content']),pid,version,effective,url,{'provenance':'shopmind_curated_experimental'}))
    return sections
def load_policy_corpus(path):
    path=Path(path); sections=load_policy_sections(path) if path.is_file() and path.suffix=='.jsonl' else _load_yaml_dir(path)
    return [KnowledgeDocument(s.citation_id,'policy',s.title,s.content,s.search_text,{'policy_id':s.policy_id,'section_id':s.section_id,'version':s.version,'effective_date':s.effective_date,'source_url':s.source_url,'policy_type':s.policy_type,**s.metadata}) for s in sections]
