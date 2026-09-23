# Baseline: untrained Qwen on riscv-formal

**Result: 0 of 99 measured problems proved.**

| core | problems | proved | how it failed |
|---|---|---|---|
| serv | 37 | **0** | BASECASE_FAIL 33, ERROR 1, NEEDS_INVARIANT 1, NO_ANSWER 2 |
| nerv | 62 | **0** | BASECASE_FAIL 59, ERROR 1, NEEDS_INVARIANT 2 |
| picorv32 | 46 | not measured | prompt 28,769 tokens + 4,000 output exceeded the 32,768 window; every request was refused by the server, so those 46 NO_ANSWER rows are a harness limit and NOT a model result |

| setting | value |
|---|---|
| date | 2026-09-22 |
| model | Qwen2.5-Coder-32B-Instruct, served locally with vLLM 0.29 |
| context | 32768 tokens |
| attempts | 1 per problem, default sampling |
| condition | native (whole design, every assertion) |
| k | 10 |
| repo commit | 8335e64 |
| raw log | initiator/logs/riscv_score.jsonl, runs 2026-09-22_17h39m51s and 2026-09-22_17h57m00s |

## The failure is falsehood, not weakness

Across the two measured cores, 33 of 37 serv
answers and 59 of 62 nerv answers were
refuted in the base case: the model asserted facts that are simply untrue of the design.
Only 3
answers in 99 were true but insufficient.

Answer length varied from 4 to 344 invariants on the same kind of problem, which reads
as guessing rather than reasoning.

This is the gap the corpus targets: every invariant in the 242 training rows is proven
true of its design by construction, so the training signal is precisely "say true things
about this design" - the thing the untrained model does not do.

## Reproduce

    export QWEN_BASE_URL=<vllm endpoint>/v1
    export QWEN_API_KEY=none QWEN_MODEL=llm
    venv/bin/python initiator/riscv_score.py --core serv
    venv/bin/python initiator/riscv_score.py --core nerv

    venv/bin/python initiator/riscv_score.py --core serv --report   # table

## Caveats, stated rather than buried

- One attempt per problem, sampling left at the server default. A pass@k
  measurement is fairer; whatever is used before training must be used after.
- picorv32 is unmeasured. Re-running it needs a longer context window, and
  because that setting changes how every prompt is read, all three cores must
  then be re-measured together for the numbers to be comparable.
- Two separate 5-problem serv runs also scored 0, so the zero is reproducible.
- Graded with yosys 0.67 and sby on macOS; the cluster has yosys 0.69.
