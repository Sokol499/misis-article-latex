"""Author-created 144 workloads for a given real model tokenizer.

Four text categories * three original stems * four input lengths * three
new-token lengths. Prompts are generated using the ACTUAL tokenizer and
input_ids are stored, avoiding ambiguous re-tokenization at run time.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

SEEDS={
    'continuation': [
        'The future of distributed computation depends on careful resource planning.',
        'An engineer inspected a sequence of tensor operations and measured its cost.',
        'Large language models can be separated into multiple ordered computing stages.',
    ],
    'qa': [
        'Question: What causes latency in a remote procedure call? Answer:',
        'Question: Why can a smaller tensor speed up communication? Answer:',
        'Question: What is the difference between throughput and latency? Answer:',
    ],
    'technical': [
        'Transformer blocks contain attention, normalization, and feed-forward operations.',
        'Each Ray actor can hold a segment of the neural network computation graph.',
        'Serialization and scheduling overhead can dominate very small model inference.',
    ],
    'instruction': [
        'Explain how to verify that model layers run in different processes.',
        'Describe a reproducible benchmark for autoregressive inference on CPU.',
        'List the assumptions behind a dynamic-programming layer partition algorithm.',
    ],
}
LENGTHS=(16,64,128,256)
OUTPUTS=(8,16,32)


def build_workloads(tokenizer, max_position_embeddings=1024):
    cases=[]
    for category, stems in SEEDS.items():
        for stem_id, stem in enumerate(stems):
            for length in LENGTHS:
                tokens=tokenizer.encode((stem+' ')*(length+4),add_special_tokens=False)
                while len(tokens)<length:
                    tokens+=tokens[:]
                actual=tokens[:length]
                for output in OUTPUTS:
                    if length+output>max_position_embeddings:
                        raise ValueError('Requested sequence exceeds model context')
                    case_id=f'{category}-{stem_id}-in{length}-out{output}'
                    cases.append({
                        'case_id':case_id,'category':category,'stem_id':stem_id,
                        'input_tokens':length,'max_new_tokens':output,
                        'prompt_text':tokenizer.decode(actual,clean_up_tokenization_spaces=False),
                        'input_ids':actual,'source':'authored seed, deterministic repetition and truncation',
                    })
    if len(cases)!=144 or len({r['case_id'] for r in cases})!=144:
        raise AssertionError('Workload specification incomplete')
    return cases


def write_workloads(path, rows, model_name):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps({'model':model_name,**row},ensure_ascii=False)+'\n')

