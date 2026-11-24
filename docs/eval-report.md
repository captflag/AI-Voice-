# Eval report: 20261124_114754

36 simulated calls by voice against `bot.py`, each on a freshly started bot with an empty appointment book.

## Task success

**36 of 36 calls passed (100%).**

| Scenario | Passed | First failure |
|---|---|---|
| greeting | 3 of 3 |  |
| book_appointment | 3 of 3 |  |
| repeated_digits | 3 of 3 |  |
| reschedule | 3 of 3 |  |
| weekend_refused | 3 of 3 |  |
| interruption | 3 of 3 |  |
| wrong_number | 3 of 3 |  |
| price_from_notes | 3 of 3 |  |
| insurance_not_accepted | 3 of 3 |  |
| parking | 3 of 3 |  |
| not_in_notes | 3 of 3 |  |
| no_medical_advice | 3 of 3 |  |

## Reply latency

From the caller's last word to the bot's first, as measured by the bot itself. The harness runs on the same machine, so these are a little slower than a real call.

| Replies | p50 | p95 | Slowest |
|---|---|---|---|
| 16 | 3451 ms | 4873 ms | 4873 ms |

| Stage | Samples | Median | p95 |
|---|---|---|---|
| Speech-to-text, per segment | 87 | 396 ms | 634 ms |
| LLM, time to first token | 123 | 557 ms | 685 ms |
| Voice, time to first audio | 58 | 820 ms | 1110 ms |

## LLM cost

200,139 input and 10,464 output tokens across all calls, about $0.0252 in total at Haiku 5.5 list price.
That is $0.0007 per call, or $0.0015 per minute of call. Speech-to-text and the voice run locally, so they add hardware cost but no per-minute fee.
