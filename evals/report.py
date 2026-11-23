"""Turn an eval run into a short report: task success, reply latency and LLM cost.

    python evals/report.py              the latest run in evals/runs
    python evals/report.py <run dir>    a specific run

Writes docs/eval-report.md and prints it.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# USD per million tokens: Claude Haiku 5.5 list price. Speech-to-text and the voice run locally and are not costed.
PRICE_IN, PRICE_OUT = 0.10, 0.50
REPLY = re.compile(r"reply started (\d+) ms after the caller stopped")


def percentile(values, q):
    """Nearest-rank percentile, so every figure reported is one that was actually measured."""
    ordered = sorted(values)
    return ordered[max(0, -(-len(ordered) * q // 100) - 1)]


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(p for p in (ROOT / "runs").iterdir() if p.is_dir())
    records = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    by_scenario, replies, stages = defaultdict(list), [], defaultdict(list)
    tokens_in = tokens_out = 0
    for r in records:
        by_scenario[r["scenario"].split("/")[-1]].append(r)
        log = Path(r["artifacts"]["log"])
        if log.exists():  # the bot's own log holds its voice-to-voice measurements
            replies += [int(ms) for ms in REPLY.findall(log.read_text(encoding="utf-8", errors="replace"))]
        for metric in r.get("bot_metrics") or []:
            data = metric["data"]
            for item in data.get("tokens", []):
                tokens_in += item["prompt_tokens"]
                tokens_out += item["completion_tokens"]
            for kind, name, label in (("processing", "WhisperSTT", "Speech-to-text, per segment"),
                                      ("ttfb", "AnthropicLLM", "LLM, time to first token"),
                                      ("ttfb", "KokoroTTS", "Voice, time to first audio")):
                stages[label] += [i["value"] * 1000 for i in data.get(kind, []) if i["processor"].startswith(name) and i["value"] > 0]

    passed = sum(r["passed"] for r in records)
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
    minutes = sum(r["duration_ms"] for r in records) / 60000
    out = [f"# Eval report: {run.name}", "",
           f"{len(records)} simulated calls by voice against `bot.py`, each on a freshly started bot with an empty appointment book.", "",
           "## Task success", "",
           f"**{passed} of {len(records)} calls passed ({passed / len(records):.0%}).**", "",
           "| Scenario | Passed | First failure |", "|---|---|---|"]
    for name, runs in by_scenario.items():
        reason = next((f["reason"] for r in runs for f in r.get("failures") or []), "")
        reason = reason.replace("|", "/")[:140]  # a bar would break the table
        out.append(f"| {name} | {sum(r['passed'] for r in runs)} of {len(runs)} | {reason} |")

    out += ["", "## Reply latency", "",
            "From the caller's last word to the bot's first, as measured by the bot itself. "
            "The harness runs on the same machine, so these are a little slower than a real call."]
    if replies:
        out += ["", "| Replies | p50 | p95 | Slowest |", "|---|---|---|---|",
                f"| {len(replies)} | {percentile(replies, 50)} ms | {percentile(replies, 95)} ms | {max(replies)} ms |"]
    out += ["", "| Stage | Samples | Median | p95 |", "|---|---|---|---|"]
    out += [f"| {label} | {len(v)} | {statistics.median(v):.0f} ms | {percentile(v, 95):.0f} ms |" for label, v in stages.items() if v]

    out += ["", "## LLM cost", "",
            f"{tokens_in:,} input and {tokens_out:,} output tokens across all calls, about ${cost:.4f} in total at Haiku 5.5 list price.",
            f"That is ${cost / len(records):.4f} per call, or ${cost / minutes:.4f} per minute of call. "
            "Speech-to-text and the voice run locally, so they add hardware cost but no per-minute fee."]

    report = "\n".join(out) + "\n"
    (ROOT.parent / "docs").mkdir(exist_ok=True)
    (ROOT.parent / "docs" / "eval-report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
