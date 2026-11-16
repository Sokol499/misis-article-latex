"""Analytical raw-payload calculator for a GPT-2-style autoregressive pipeline.

Not a network monitor and not a Ray benchmark. Python stdlib only.
Run: python3 traffic_model.py
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    prompt_tokens: int = 5
    generated_tokens: int = 20
    vocab_size: int = 50257
    batch_size: int = 1
    logit_bytes: int = 4  # float32
    token_id_bytes: int = 8  # int64

    def __post_init__(self):
        if min(vars(self).values()) <= 0:
            raise ValueError("All scenario fields must be positive")


def raw_output_payloads(s: Scenario) -> dict[str, int]:
    """Bytes sent from last shard to controller (not network measurements).

    D0: full [B,L,V] logits per step
    D1: calculate full logits but return [B,V] at last position
    D2: calculate only last-position logits and return [B,V]
    D3: final shard calculates argmax and returns [B] int64 token IDs
    """
    sum_context_lengths = (
        s.generated_tokens * s.prompt_tokens
        + s.generated_tokens * (s.generated_tokens - 1) // 2
    )
    full_logits = s.batch_size * s.vocab_size * s.logit_bytes * sum_context_lengths
    last_logits = s.batch_size * s.vocab_size * s.logit_bytes * s.generated_tokens
    token_ids = s.batch_size * s.token_id_bytes * s.generated_tokens
    return {"D0": full_logits, "D1": last_logits, "D2": last_logits, "D3": token_ids}


def human_mib(nbytes: int) -> str:
    return f"{nbytes / 2**20:.4f} MiB"


def report(s: Scenario) -> str:
    values = raw_output_payloads(s)
    rows = [f"Scenario: L0={s.prompt_tokens}, T={s.generated_tokens}, V={s.vocab_size}, B={s.batch_size}"]
    for key, val in values.items():
        rows.append(f"{key}: {val:,} bytes ({human_mib(val)})".replace(",", " "))
    rows.append(f"Potential reduction D0 -> D3: {values['D0'] / values['D3']:,.1f}x in modeled return payload".replace(",", " "))
    rows.append("CAUTION: Ray serialization, RPC, hidden states, masks and transport overhead are excluded.")
    return "\n".join(rows)


if __name__ == "__main__":
    print(report(Scenario()))
