# Baseline: untrained Qwen on riscv-formal

**Result: 0 of 37 proved.**

| setting | value |
|---|---|
| date | 2026-09-22 |
| model | Qwen2.5-Coder-32B-Instruct, served locally with vLLM 0.29 |
| context | 32768 tokens |
| attempts | 1 per problem, default sampling |
| core | serv, 674 lines |
| condition | native (whole design, every assertion) |
| problem set | the 37 serv checks the survey found unprovable by k-induction alone |
| k | 10 |
| repo commit | 1ddd1af |
| raw log | initiator/logs/riscv_score.jsonl, run_id 2026-09-22_17h39m51s |

## How it failed

| verdict | count | meaning |
|---|---|---|
| BASECASE_FAIL | 33 | the invariants are false of the core; refuted immediately |
| NEEDS_INVARIANT | 1 | true but not strong enough to close the proof |
| NO_ANSWER | 2 | nothing usable returned |
| ERROR | 1 | malformed expressions (`==>`, which is not Verilog) |

The dominant failure is not weakness but falsehood: 33 of
37 answers asserted something untrue of the design. Answer sizes ranged from
0 to 344 invariants (one answer gave 344), which reads as guessing
rather than reasoning.

## Reproduce

    export QWEN_BASE_URL=<vllm endpoint>/v1
    export QWEN_API_KEY=none QWEN_MODEL=llm
    venv/bin/python initiator/riscv_score.py --core serv

## Caveats, stated rather than buried

- One attempt per problem. A pass@k measurement is the fairer comparison and
  is not yet done; whatever is used before training must be used after.
- Sampling was left at the server default, not pinned.
- Two earlier 5-problem runs on the same day scored 0/5 each, so the zero is
  reproducible rather than a single unlucky draw.
- serv only. nerv (62 problems) and picorv32 (46) are not yet measured.
