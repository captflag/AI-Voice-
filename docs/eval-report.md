# Eval report: 20261123_170240

21 simulated calls by voice against `bot.py`, each on a freshly started bot with an empty appointment book.

## Task success

**17 of 21 calls passed (81%).**

| Scenario | Passed | First failure |
|---|---|---|
| greeting | 3 of 3 |  |
| book_appointment | 3 of 3 |  |
| repeated_digits | 0 of 3 | function call 'book_appointment' not seen (matched: none) |
| reschedule | 3 of 3 |  |
| weekend_refused | 2 of 3 | text '' does not match '(?i)closed/not open/weekday/Monday (to/through) Friday' |
| interruption | 3 of 3 |  |
| wrong_number | 3 of 3 |  |

## Reply latency

From the caller's last word to the bot's first, as measured by the bot itself. The harness runs on the same machine, so these are a little slower than a real call.

| Replies | p50 | p95 | Slowest |
|---|---|---|---|
| 16 | 3531 ms | 4583 ms | 4583 ms |

| Stage | Samples | Median | p95 |
|---|---|---|---|
| Speech-to-text, per segment | 75 | 360 ms | 696 ms |
| LLM, time to first token | 98 | 581 ms | 746 ms |
| Voice, time to first audio | 39 | 851 ms | 1084 ms |

## LLM cost

107,230 input and 7,769 output tokens across all calls, about $0.0146 in total at Haiku 5.5 list price.
That is $0.0007 per call, or $0.0010 per minute of call. Speech-to-text and the voice run locally, so they add hardware cost but no per-minute fee.
