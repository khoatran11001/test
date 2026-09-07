from typing import Protocol
def answer_correctness(answer,acceptable_facts,expected_insufficient,predicted_insufficient):
    if expected_insufficient: return float(predicted_insufficient)
    if not acceptable_facts: return 1.0
    lowered=answer.casefold(); return sum(f.casefold() in lowered for f in acceptable_facts)/len(acceptable_facts)
class GroundednessGrader(Protocol):
    def grade(self,*,question:str,answer:str,context_ids:tuple[str,...])->tuple[float,int]: ...
