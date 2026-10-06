"""The benchmark authors' repair loop, on our scorer.

Large Lemma Miners (Peled et al., IJCAI 2026) publish an agentic mode,
src/agentic.py with src/evaluation.py, and we copy its protocol so a
repaired score means what theirs means:

  - up to `num_iterations=5` model answers in all, the first included
    (agentic.py:33, :153);
  - every answer's lemmas judged one by one with three EBMC checks
    (evaluation.py:349-360): no counterexample within 30 steps from reset
    (`correctness_bounded`, :590), 1-inductive alone (`one_induction`,
    :554) and 1-inductive with the property (`one_inductive_with_prop`,
    :563); the two induction checks retry with --buechi on ERROR;
  - a lemma is correct if it is 1-inductive, or has no 30-step
    counterexample and is not refuted in one step (:330-334);
  - if no single lemma closes the proof, the correct ones are tried
    together, 1-inductive first then by text, one more at a time, until
    the set closes it (:365-411);
  - correct lemmas are kept across answers, and before every new answer
    the kept set is tried again (agentic.py:191-210, :298-321);
  - solved means some set is 1-inductive with the property (:343-347);
  - each lemma gets their feedback sentence (agentic.py:226-283), with
    their repair template on the 1st, 3rd and 5th feedback (:236);
  - a reply with no answer gets their format message (:85-101).

Differences, each on purpose:
  - Answer 1 is our own question and one-shot grading
    (benchmark_solve.PROMPT), so round 0 reproduces our one-shot scores.
  - Lemmas are plain expressions in {"invariants": [...]}, wrapped by
    ebmc_eval.wrap_lemmas, not their <json> property blocks; the format
    parts of their template and format message ask for ours.
  - The reset is scoped to the file's own module (ebmc_eval.top_module),
    as our one-shot scorer already does; their `--reset rst` fails on the
    hard set.
  - None on the EBMC limit: the per-lemma checks use theirs, 120 s
    (scripts/experiment_agentic.py:30). Only round 0's one-shot grade, in
    benchmark_solve, keeps our 300 s, so it still matches the earlier
    one-shot scores. (300 s here made a single answer of 16 lemmas take
    an hour on lrg_arb_lrg_16, 6 Oct 2026.)
  - Counterexample traces are capped at TRACE_CAP characters so five
    answers fit the 32,768-token window, and the loop stops cleanly
    ("context_full") when they no longer would.
  - When kept lemmas are tried together, only the check that decides
    "solved" is run on each set, and a one-lemma set reuses its
    singleton result; their code reruns all three, which changes nothing
    but time.
  - A lemma shown twice in one feedback (their prefix re-check lists the
    first kept lemma again) is shown once.

The control (`give_feedback=False`), ours, not theirs: the same loop with
the feedback taken out, to tell whether a repaired score comes from what
the feedback says or from the extra answers. Every lemma is still judged,
kept and tried together, the model still sees its earlier answers, and an
unreadable reply still gets the format message; but between answers the
model hears only NO_FEEDBACK, never which lemmas were right or wrong.
"""
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ebmc_eval
from solver_baseline import parse_invariants

ROUNDS = 5                  # their num_iterations, the first answer included
TRACE_CAP = 4000            # characters of counterexample shown per lemma
WINDOW = 32768              # the server's max model length
MAX_TOKENS = 16000          # our one-shot reply budget
MIN_REPLY = 1000            # below this, stop rather than ask for a stub
FACT_TIMEOUT_S = 120        # their EBMC limit per check (experiment_agentic.py:30)

# repair_template.txt, verbatim up to its Output Format section
REMINDER_HEAD = ("You are given feedback for your lemmas below. Read it "
                 "carefully, it is meant to guide you towards giving better "
                 "lemmas.")
REMINDER = REMINDER_HEAD + """

## Reminder
- Your task is to generate lemmas such that, when combined with the target safety property, the conjunction is 1-inductive.
- The target safety property remains the same.
- You have already seen the SystemVerilog module in the first message of this conversation.

Below are the candidate lemmas from the last iteration and feedback for each.

## Instructions
- If a lemma is incorrect, revise it or remove it altogether.
- If a lemma is correct but fails to form a one-inductive argument when combined with the property, revise it or propose an alternative lemma that does.
- You may also introduce new lemmas that are likely to help.
- Keep the list concise and relevant.

## Output Format
- Please stick to the output format that was specified in the first message. I.e., give the proposed lemmas as JSON: {"invariants": ["<expr>", ...]}


## Feedback for previous lemmas
"""

FORMAT_REMINDER = 'Provide the proposed lemmas as JSON: {"invariants": ["<expr>", ...]}'

# what they send when there is no feedback to give (agentic.py:290-293)
EMPTY_FEEDBACK = ("Please propose new lemmas. Follow the JSON format "
                  "specified earlier.")
# the control's message: theirs, plus "not solved yet", which the feedback
# version tells the model too
NO_FEEDBACK = "Your lemmas do not prove the property yet. " + EMPTY_FEEDBACK


# --- one EBMC run, read the authors' way --------------------------------

def _trace(stdout):
    m = re.search(r"Counterexample:\s*\n(?:\s*\n)*(.+?)(?:\Z)", stdout, re.S)
    return m.group(1).strip() if m else ""


def ebmc_result(stdout, stderr, mode):
    """evaluation.py run_ebmc's reading of EBMC's output (:727-780). Its
    order matters: "PROVED" is tested first, so a 30-step check that
    finds nothing ("PROVED up to bound 30") counts as PROVEN."""
    pattern = r"^\*\* Results:.* PROVED\s*$" if mode == "correctness" \
        else "PROVED"
    if re.search(pattern, stdout, re.S | re.M):
        return {"verdict": "PROVEN"}
    if "REFUTED" in stdout:
        return {"verdict": "CEX", "info": _trace(stdout) or "No trace extracted"}
    if "INCONCLUSIVE" in stdout:
        return {"verdict": "INCONCLUSIVE"}
    if "FAILURE:" in stdout or "Assertion failure" in stdout:
        m = re.search(r"line \d+: (.*)", stdout, re.S)
        return {"verdict": "ERROR", "info": m.group(1) if m else ""}
    if stderr:
        return {"verdict": "ERROR", "info": stderr}
    return {"verdict": "TIMEOUT"}


def run_mode(bench_path, lemmas, mode, workdir, buechi=False):
    source = Path(bench_path).read_text()
    out = Path(workdir) / f"{Path(bench_path).stem}_{mode}_{uuid.uuid4().hex[:8]}.sv"
    out.write_text(ebmc_eval.build_variant(source, lemmas, mode))
    cmd = ebmc_eval.ebmc_command(out, mode,
                                 rst=bool(ebmc_eval._RST.search(source)),
                                 ebmc=ebmc_eval.EBMC, source=source)
    if buechi:
        cmd += " --buechi"
    t0 = time.perf_counter()
    try:
        import subprocess
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           errors="replace", timeout=FACT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "time": FACT_TIMEOUT_S}
    finally:
        out.unlink(missing_ok=True)
    res = ebmc_result(r.stdout, r.stderr, mode)
    res["time"] = round(time.perf_counter() - t0, 2)
    return res


# --- one set of lemmas: their three checks and booleans -----------------

def judge(bench_path, lemmas, workdir, run=run_mode):
    """_get_all_verification_results: the three checks on one set."""
    lemmas = list(lemmas)

    def with_buechi(mode):
        r = run(bench_path, lemmas, mode, workdir)
        if r["verdict"] == "ERROR":
            return run(bench_path, lemmas, mode, workdir, buechi=True)
        return r
    return {"lemma": lemmas,
            "correct": run(bench_path, lemmas, "correctness_bounded", workdir),
            "1-inductive": with_buechi("one_induction"),
            "1-inductive with property": with_buechi("one_inductive_with_prop")}


def booleans(e):
    """translate_entry_to_booleans (:302-340)."""
    v = lambda key: e[key]["verdict"]                          # noqa: E731
    return {"correct": v("1-inductive") == "PROVEN"
            or (v("correct") == "PROVEN" and v("1-inductive") != "CEX"),
            "1-inductive": v("1-inductive") == "PROVEN",
            "1-inductive with property":
                v("1-inductive with property") == "PROVEN",
            "error": v("correct") == "ERROR"}


def solves(e):
    return booleans(e)["1-inductive with property"]


def evaluate_subsets(bench_path, lemmas, workdir, run=run_mode, workers=4):
    """Singletons, then the correct ones together (evaluate_subsets)."""
    lemmas = list(dict.fromkeys(lemmas))
    if not lemmas:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        singles = list(ex.map(
            lambda l: judge(bench_path, [l], workdir, run), lemmas))
    if any(solves(e) for e in singles):
        return singles
    ranked = sorted((e for e in singles if booleans(e)["correct"]),
                    key=lambda e: (not booleans(e)["1-inductive"], e["lemma"][0]))
    out = list(singles)
    for i in range(1, len(ranked) + 1):
        prefix = [e["lemma"][0] for e in ranked[:i]]
        if i == 1:
            res = dict(ranked[0])
        else:
            skipped = {"verdict": "NOT_RUN"}
            res = {"lemma": prefix, "correct": skipped, "1-inductive": skipped,
                   "1-inductive with property": run(
                       bench_path, prefix, "one_inductive_with_prop", workdir)}
            if res["1-inductive with property"]["verdict"] == "ERROR":
                res["1-inductive with property"] = run(
                    bench_path, prefix, "one_inductive_with_prop", workdir,
                    buechi=True)
        out.append(res)
        if solves(res):
            break
    return out


# --- the feedback ---------------------------------------------------------

def _cap(text):
    if len(text) <= TRACE_CAP:
        return text
    return text[:TRACE_CAP] + f"\n... (trace cut at {TRACE_CAP} characters)"


def _extract_error(text):
    m = re.search(r"(line \d+:.*?)(?:CONVERSION ERROR|$)", text,
                  re.S | re.I)
    return m.group(1).strip() if m else text.strip()


def repair_message(singletons, counter, show_cex):
    """_repair_message (agentic.py:226-283). `counter` is the number of
    answers before the one this feedback is about."""
    message = []
    if not counter % 2:
        message.extend(REMINDER.splitlines(keepends=True))
    seen = set()
    for e in singletons:
        lemma = e["lemma"][0]
        if lemma in seen:
            continue
        seen.add(lemma)
        message.append(f"\n// Feedback for {lemma}: ")
        correct = e["correct"]
        if booleans(e)["1-inductive with property"]:
            message.append("The conjunction of the lemma and the property is "
                           "1-inductive, so the lemma helps prove the "
                           "property. Good job!")
        elif correct["verdict"] == "PROVEN":
            message.append("The lemma is correct.")
            message.append("However, it does not form a 1-inductive argument "
                           "when conjoined with the property. Please give a "
                           "better lemma.")
        elif correct["verdict"] == "CEX":
            if show_cex:
                message.append("The lemma is incorrect. Here is a "
                               "counterexample: ")
                message.append(_cap(correct.get("info", "")))
            else:
                message.append("The lemma is incorrect. Repair or replace it.")
        elif correct["verdict"] == "ERROR":
            message.append("There's an error with the lemma: ")
            message.append(_extract_error(correct.get("info", "")))
        else:
            # TIMEOUT; their code raises on anything else, which a bounded
            # check cannot return
            message.append("The lemma timed out and its correctness could "
                           "not be determined.")
    text = "\n".join(message)
    if not text.strip():
        return EMPTY_FEEDBACK
    return text


def no_json_message(info):
    """_no_json_block_message, asking for our format."""
    return ("Your last response did not follow the expected format. It "
            f"yielded the following error: {info}" + FORMAT_REMINDER)


def max_tokens_for(messages):
    """Room left for a reply in the server's window, from a rough count
    (about 3 characters a token for Verilog). None when under MIN_REPLY."""
    estimate = sum(len(m["content"]) for m in messages) // 3 \
        + 8 * len(messages)
    room = WINDOW - estimate - 512
    return None if room < MIN_REPLY else min(MAX_TOKENS, room)


# --- the conversation -----------------------------------------------------

def _per_lemma(singletons):
    rows, seen = [], set()
    for e in singletons:
        lemma = e["lemma"][0]
        if lemma in seen:
            continue
        seen.add(lemma)
        b = booleans(e)
        info = e["correct"].get("info")
        rows.append({"lemma": lemma, "correct": b["correct"],
                     "one_inductive": b["1-inductive"],
                     "inductive_with_prop": b["1-inductive with property"],
                     "error": b["error"],
                     "correctness": e["correct"]["verdict"],
                     "one_induction": e["1-inductive"]["verdict"],
                     "with_prop": e["1-inductive with property"]["verdict"],
                     "info": _cap(info)[:600] if info else None})
    return rows


def remembered(run):
    """Their evaluator caches every EBMC query (evaluation.py:413-430), so
    re-trying the kept lemmas each round costs nothing; this does the same
    for one file's conversation."""
    import threading
    memo, lock = {}, threading.Lock()

    def cached(bench_path, lemmas, mode, workdir, buechi=False):
        key = (tuple(lemmas), mode, buechi)
        with lock:
            if key in memo:
                return memo[key]
        r = run(bench_path, lemmas, mode, workdir, buechi=buechi)
        with lock:
            memo[key] = r
        return r
    return cached


def repair(bench_path, question, first_reply, first_lemmas, ask, *,
           rounds=ROUNDS, show_cex=False, workdir, run=None, log,
           workers=4, give_feedback=True):
    """The loop after our first answer. `ask(messages, max_tokens)` returns
    (text, finish). `log(row)` is called once per answer, round 0 first.
    `give_feedback=False` runs the control (see the module docstring)."""
    run = remembered(run or run_mode)
    messages = [{"role": "user", "content": question}]
    kept, proposed = [], set()
    solved, solved_round, stop = False, None, None
    reply, finish, lemmas = first_reply, None, first_lemmas
    answer_round, counter, wall = 0, 0, 0.0
    while True:
        t0 = time.monotonic()
        singletons = []
        if lemmas is None:
            feedback = no_json_message("no JSON object with an invariants "
                                       "list in the reply")
        else:
            entries = evaluate_subsets(bench_path, lemmas, workdir, run,
                                       workers)
            singletons = [e for e in entries if len(e["lemma"]) == 1]
            solved = solved or any(solves(e) for e in entries)
            for e in singletons:
                lemma = e["lemma"][0]
                if lemma in proposed:
                    continue
                proposed.add(lemma)
                if booleans(e)["correct"]:
                    kept.append(lemma)
            feedback = repair_message(singletons, counter, show_cex) \
                if give_feedback else NO_FEEDBACK
        counter += 1
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": feedback}]
        if not solved and kept:
            solved = any(solves(e) for e in evaluate_subsets(
                bench_path, kept, workdir, run, workers))
        if solved and solved_round is None:
            solved_round = answer_round
        if solved:
            stop = "solved"
        elif counter >= rounds:
            stop = "max_iterations"
        room = None if stop else max_tokens_for(messages)
        if not stop and room is None:
            stop = "context_full"
        row = {"round": answer_round, "raw": reply, "finish": finish,
               "lemmas": lemmas, "feedback": feedback,
               "per_lemma": _per_lemma(singletons),
               "kept": list(kept), "solved": solved,
               "solve_wall_s": round(wall + time.monotonic() - t0, 2)}
        if stop:
            row.update(final=True, stop_reason=stop,
                       solved_round=solved_round)
            log(row)
            break
        log(row)
        t1 = time.monotonic()
        try:
            reply, finish = ask(messages, room)
        except Exception as exc:
            text = str(exc).lower()
            stop = "context_full" if "context length" in text \
                or "maximum context" in text else "prompt_error"
            log({"round": answer_round + 1, "raw": None, "finish": None,
                 "lemmas": None, "per_lemma": [], "kept": list(kept),
                 "solved": solved, "solve_wall_s": 0.0, "final": True,
                 "stop_reason": stop, "solved_round": solved_round,
                 "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        wall = time.monotonic() - t1
        lemmas = parse_invariants(reply or "")
        answer_round += 1
    return {"solved": solved, "solved_round": solved_round,
            "stop_reason": stop, "kept": kept, "answers": counter}
