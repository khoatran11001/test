from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class EfficiencyMetrics:
    device:str; warmup_batches:int; timed_batches:int; items:int; total_seconds:float; items_per_second:float; p50_batch_latency_seconds:float; p95_batch_latency_seconds:float; peak_cuda_bytes:int|None
    def to_dict(self):return asdict(self)
def aggregate_batch_timings(*,batch_durations,items_per_batch,total_seconds,peak_cuda_bytes,device,warmup_batches=0):
    if not batch_durations or len(batch_durations)!=len(items_per_batch): raise ValueError('timings must be aligned and non-empty')
    a=np.asarray(batch_durations,dtype=float); items=sum(items_per_batch)
    return EfficiencyMetrics(device,warmup_batches,len(a),items,float(total_seconds),items/total_seconds if total_seconds>0 else 0.0,float(np.quantile(a,.5)),float(np.quantile(a,.95)),peak_cuda_bytes)
