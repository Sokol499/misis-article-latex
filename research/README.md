# Experimental practice for PZ-4 — distributed Transformer inference

This is a **new independent, auditable implementation** consistent with the NIR6 architecture. The original `distributed-llm-ray-nir` source code was not supplied to this workspace; no claim is made that this modifies the original project.

## What is implemented

- Real CPU inference with Hugging Face `sshleifer/tiny-gpt2` using Ray remote actors. Shards execute different contiguous GPT-2 blocks in distinct processes. The default GitHub Actions environment is **one physical VM and one Ray node**, not a three-node physical cluster.
- Monolithic B0 and distributed D0, D1, D2, D3 variants. D0 returns `[B,L,V]` logits; D1 computes all logits but returns `[B,V]`; D2 computes last-position logits only; D3 computes last-position `argmax` inside the last actor and returns `int64` IDs.
- A deterministic, independently authored set of 144 workload definitions (4 categories × 3 stems × 4 input lengths × 3 output lengths). Source prompts are constructed by repetition/truncation at token-ID level, NOT independent natural-language prompts; effective diversity is 12 stems. Not a model-training dataset.
- Strict token-level verification versus monolithic greedy decoding; mismatches are recorded and fail the run.
- Actual wall-clock timings on the running CPU, plus the *raw logical payload bytes* passed back by the last stage, deliberately distinguished from measured NIC traffic.
- Exact dynamic programming for ordered layer partition with per-stage compute times, memory limits and per-boundary communication costs; tested against brute force. The built-in example uses explicitly **synthetic cost data**, not a claimed measured speedup on Mac or external servers.

## Running

Python 3.11 is recommended. On a machine with Python, pip and enough disk space:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu  # Linux CPU runner
python -m pip install -r research/requirements.txt
python -m unittest discover -s research/tests -v
python -m research.runner --model sshleifer/tiny-gpt2 --limit 4 --repeats 2 --output research/results/pilot
python -m research.runner --model sshleifer/tiny-gpt2 --limit 0 --repeats 3 --output research/results/full
```

On Apple Silicon macOS, install a compatible PyTorch from the official instructions instead of the Linux-only CPU index; Ray workers use CPU, not Metal/MPS. Start with pilot runs and monitor memory before running all 144.

The `--limit 4` option runs only four of 144 cases; `--limit 0` runs the full matrix. The four first scenarios share the same source text and are **not representative**; a full experiment must examine all categories, lengths and variability. `environment.json` records version, logical nodes and actor PIDs; `measurements.csv` has times and correctness; `mismatches.json` preserves discrepancies; `workloads.jsonl` contains all 144 workloads. The full study needs repetitions, fixed local hardware and critical review of the resulting measurements.

Run the partition optimizer and compare to exhaustive enumeration:

```sh
python - <<'PY'
from research.partition import synthetic_case, optimal_partition, brute_force_partition
s = synthetic_case(12, 3)
print('DP:', optimal_partition(*s))
print('Brute force:', brute_force_partition(*s))
PY
```

## Methodological limitations

1. Original NIR tests and times were on a different host: no implied Mac timing equivalence.
2. Without KV-cache, every new token recomputes the complete context. A larger model may be extremely slow.
3. The synthetic partition-cost example proves mathematical correctness *for that cost model*, not real optimality; measured per-stage profiles must replace it for real deployment.
4. The observed returned-array `nbytes` is not true TCP/IP, Ray serialization or physical Ethernet traffic.
5. Run on a *single-node Ray process cluster* by default; the original three Docker Ray nodes are a separate existing NIR result and this module must not be mislabeled as equivalent without additional execution.
6. Report failed cases, exhausted memory and token mismatches instead of silently excluding them.
7. A genuine experiment needs artifacts of raw timings and platform information, not just analytical estimates.

For the academic paper, see the parent repository. Habr posts must be written and published by their human author in accordance with the platform's current authoring rules.


See [measured results and limitations](RESEARCH_EVIDENCE.md) and [author notes](HABR_AUTHOR_NOTES.md). The source runs are in [evidence/run_artifacts.zip](evidence/run_artifacts.zip).
