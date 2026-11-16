"""Resource-constrained partition of ordered Transformer layers.

The mathematical objective is additive single-request predicted latency;
it is NOT a measurement nor a pipeline-throughput optimum.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import inf


@dataclass(frozen=True)
class Plan:
    boundaries: tuple[int, ...]
    predicted_ms: float
    memory_mib: tuple[float, ...]


def evaluate(bounds, compute_ms, memory_mib, limits_mib, link_ms=None, fixed_mem_mib=None):
    """Return (predicted cost, per-stage memory) or (inf, ...) if invalid.

    compute_ms[stage][layer] are profiled or hypothetical layer execution times.
    link_ms[stage][boundary] is cost crossing stage->next at boundary.
    fixed_mem_mib can account for embeddings and output head per stage.
    """
    k, n = len(compute_ms), len(memory_mib)
    fixed = fixed_mem_mib or [0.0] * k
    links = link_ms or [[0.0] * (n + 1) for _ in range(k - 1)]
    if len(bounds) != k + 1 or bounds[0] != 0 or bounds[-1] != n or any(b <= a for a,b in zip(bounds, bounds[1:])):
        return inf, ()
    if len(limits_mib) != k or len(fixed) != k or len(links) != k - 1 or any(len(row) != n for row in compute_ms) or any(len(row) < n + 1 for row in links):
        raise ValueError('Invalid dimensions')
    mems = tuple(sum(memory_mib[bounds[i]:bounds[i+1]])+fixed[i] for i in range(k))
    if any(m > lim + 1e-10 for m,lim in zip(mems,limits_mib)):
        return inf, mems
    cost = sum(sum(compute_ms[i][bounds[i]:bounds[i+1]]) for i in range(k))
    cost += sum(links[i][bounds[i+1]] for i in range(k - 1))
    return cost, mems


def optimal_partition(compute_ms, memory_mib, limits_mib, link_ms=None, fixed_mem_mib=None):
    """Optimal ordered nonempty partition in O(K*N^2), for the additive model."""
    k, n = len(compute_ms), len(memory_mib)
    if not 1 <= k <= n:
        raise ValueError('Require 1 <= number of stages <= number of layers')
    fixed = list(fixed_mem_mib or [0.0]*k)
    links = link_ms or [[0.0]*(n+1) for _ in range(k-1)]
    if len(limits_mib) != k or len(fixed) != k or len(links) != k-1 or any(len(r)!=n for r in compute_ms) or any(len(r)<n+1 for r in links):
        raise ValueError('Invalid dimensions')
    cp = [[0.0] for _ in range(k)]
    for i in range(k):
        for x in compute_ms[i]:
            cp[i].append(cp[i][-1] + x)
    mp=[0.0]
    for x in memory_mib: mp.append(mp[-1]+x)
    dp=[[inf]*(n+1) for _ in range(k+1)]
    back=[[None]*(n+1) for _ in range(k+1)]
    dp[0][0]=0.0
    for stage in range(1,k+1):
        idx=stage-1
        for end in range(stage,n-(k-stage)+1):
            for start in range(stage-1,end):
                if mp[end]-mp[start]+fixed[idx] > limits_mib[idx]+1e-10: continue
                if dp[stage-1][start] == inf: continue
                comm=links[idx-1][start] if idx else 0.0
                cost=dp[stage-1][start]+cp[idx][end]-cp[idx][start]+comm
                if cost < dp[stage][end]-1e-12:
                    dp[stage][end]=cost
                    back[stage][end]=start
    if dp[k][n] == inf: return None
    bounds=[n]
    end=n
    for stage in range(k,0,-1):
        end=back[stage][end]
        bounds.append(end)
    bounds=tuple(reversed(bounds))
    score, mem = evaluate(bounds,compute_ms,memory_mib,limits_mib,links,fixed)
    return Plan(bounds,score,mem)


def brute_force_partition(compute_ms, memory_mib, limits_mib, link_ms=None, fixed_mem_mib=None):
    k,n=len(compute_ms),len(memory_mib)
    best=None
    for cuts in combinations(range(1,n),k-1):
        boundaries=(0,*cuts,n)
        val, mem=evaluate(boundaries,compute_ms,memory_mib,limits_mib,link_ms,fixed_mem_mib)
        if val!=inf and (best is None or val<best.predicted_ms-1e-12):
            best=Plan(boundaries,val,mem)
    return best


def synthetic_case(n=12,k=3):
    """Deterministic *synthetic* heterogeneity; not profiled from hardware."""
    speeds=[1.0,1.8,0.7,1.3]
    compute=[[((j%5)+3)*speeds[i] for j in range(n)] for i in range(k)]
    memory=[24.0+5.0*(j%3) for j in range(n)]
    limits=[float(200 + 50*i) for i in range(k)]
    links=[[3.0+0.02*j for j in range(n+1)] for i in range(k-1)]
    return compute,memory,limits,links

