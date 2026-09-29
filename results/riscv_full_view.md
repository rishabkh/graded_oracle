# Full view: Opus on serv, shown everything the prover reads

**Result: 0 of 10 proved, and 0 false claims. Shown only the wiring and not the property, 8 of 10 answers contained a false claim. Shown everything the proof tool reads, none did.**

| | before | after |
|---|---|---|
| what the model was shown | `serv_top.v` only (wire declarations and sub-module hookups, no logic) and the check's top file (four `include` lines, no assertion) | every file the proof tool reads for the check: 16 design files and 5 or 6 check files, minus the generated macro library |
| a claim contradicted by a real run (BASECASE_FAIL) | 8 | **0** |
| every claim held, but the proof did not close (NEEDS_INVARIANT) | 0 | **6** |
| ran out of output budget (NO_ANSWER) | 2 | 4 |
| proved | 0 | 0 |

Only the prompt changed between the two runs. The model, route, budget, thinking level, problems and checker were the same.

## Per problem

| check | before | after |
|---|---|---|
| causal_ch0 | BASECASE_FAIL, 20 claims | NEEDS_INVARIANT, 16 claims |
| insn_add_ch0 | NO_ANSWER (budget) | NEEDS_INVARIANT, 17 claims |
| insn_addi_ch0 | BASECASE_FAIL, 15 claims | NEEDS_INVARIANT, 12 claims |
| insn_and_ch0 | BASECASE_FAIL, 25 claims | NO_ANSWER (budget) |
| insn_andi_ch0 | BASECASE_FAIL, 29 claims | NO_ANSWER (budget) |
| insn_auipc_ch0 | BASECASE_FAIL, 29 claims | NO_ANSWER (budget) |
| insn_beq_ch0 | BASECASE_FAIL, 13 claims | NEEDS_INVARIANT, 15 claims |
| insn_bgeu_ch0 | BASECASE_FAIL, 20 claims | NEEDS_INVARIANT, 20 claims |
| insn_blt_ch0 | NO_ANSWER (budget) | NO_ANSWER (budget) |
| insn_bltu_ch0 | BASECASE_FAIL, 31 claims | NEEDS_INVARIANT, 19 claims |

Every NO_ANSWER, before and after, hit the 32,000-token limit while still thinking.

## Settings

| setting | value |
|---|---|
| model | Claude Opus 5, through OpenRouter (`anthropic/claude-opus-5`) |
| reasoning effort | high |
| output budget | 32,000 tokens |
| attempts | 1 per problem |
| k | 10 |
| problems | the first 10 serv checks that need a strengthening invariant, in name order |
| before | condition `native`, run 2026-09-26_10h29m47s |
| after | condition `full`, run 2026-09-28_17h50m59s |
| prompt size | about 10,700 tokens before, 58,000 to 64,000 after |
| cost of the after run | $9.56 at list price, plus the OpenRouter fee |
| raw log | `initiator/logs/riscv_score.jsonl` (kept on disk; not tracked in git) |

## What the verdicts mean

- **BASECASE_FAIL**: a real run of the core, starting from reset, broke one of the claims (or the property) within 10 cycles.
- **NEEDS_INVARIANT**: no claim and not the property broke on any run in those 10 cycles, but k-induction still did not close. True as far as checked; not enough to prove.
- **NO_ANSWER**: the model used its whole budget and returned nothing usable.

A NEEDS_INVARIANT means the start-up check ran to completion: the proof tool stops at once when the start-up check fails, but when only the induction step fails it waits for the start-up check to finish (`sby_engine_smtbmc.py`, `last_exit_callback`). Consistent with this, scored serv runs take about 0.9 s when the start-up check fails and about 2.1 s when it passes.

## Reading

1. **The false claims came from what the model was not shown.** With the design's logic and the property in view: 6 answers, 99 claims, none broken by any run.
2. **The remaining gap is "not enough", not "untrue".** That is the gap a repair loop targets: each NEEDS_INVARIANT comes with a counterexample showing a state the claims fail to rule out.
3. **Running out of budget rose from 2 to 4.** The prompt is about six times longer, and the model thinks longer. Four answers cannot be scored. A larger budget on those four would be a separate test, changing one thing.

## Caveats

- 10 problems, one attempt each, one core, one model.
- All earlier riscv-formal results (untrained Qwen 0 of 99 in `baseline_riscv.md`, Opus 0 of 5, Opus with full thinking 0 of 10) were measured with the `native` prompt, which hid the assertion on every core and the logic on serv. `baseline_riscv.md` describes that condition as "whole design, every assertion", which is inaccurate.
- The name check added with this condition (answers naming a signal the injection module does not declare become OUT_OF_SCOPE) did not fire.

## Reproduce

    hwtools
    export QWEN_BASE_URL=https://openrouter.ai/api/v1
    export QWEN_API_KEY=<openrouter key> QWEN_MODEL=anthropic/claude-opus-5
    venv/bin/python initiator/riscv_score.py --core serv --n 10 --condition full \
      --max-tokens 32000 --reasoning-effort high
    venv/bin/python initiator/riscv_score.py --report
