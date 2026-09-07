from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference:float; lower:float; upper:float; confidence:float; samples:int
def paired_bootstrap_difference(a,b,*,seed=20260907,samples=2000,confidence=.95):
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    if x.shape!=y.shape or x.ndim!=1 or len(x)==0: raise ValueError('paired samples must be non-empty and equal length')
    diff=x-y; rng=np.random.default_rng(seed); means=np.empty(samples)
    for i in range(samples): means[i]=diff[rng.integers(0,len(diff),size=len(diff))].mean()
    alpha=(1-confidence)/2
    return BootstrapDifference(float(diff.mean()),float(np.quantile(means,alpha)),float(np.quantile(means,1-alpha)),confidence,samples)
BootstrapInterval = BootstrapDifference
def paired_bootstrap_delta(a,b,*,seed,samples,confidence):
    r=paired_bootstrap_difference(a,b,seed=seed,samples=samples,confidence=confidence)
    return type('BootstrapInterval',(object,),{'mean_delta':r.mean_difference,'lower':r.lower,'upper':r.upper,'samples':samples,'confidence':confidence,'seed':seed,'__eq__':lambda self,o:self.__dict__==o.__dict__})()
