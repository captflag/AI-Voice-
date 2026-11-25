# Eval report: 20261125_141330

13 simulated calls by voice against `bot.py`, each on a freshly started bot with an empty appointment book.

## Task success

**13 of 13 calls passed (100%).**

| Scenario | Passed | First failure |
|---|---|---|
| greeting | 1 of 1 |  |
| book_appointment | 1 of 1 |  |
| repeated_digits | 1 of 1 |  |
| reschedule | 1 of 1 |  |
| weekend_refused | 1 of 1 |  |
| interruption | 1 of 1 |  |
| wrong_number | 1 of 1 |  |
| message_when_nobody_free | 1 of 1 |  |
| price_from_notes | 1 of 1 |  |
| insurance_not_accepted | 1 of 1 |  |
| parking | 1 of 1 |  |
| not_in_notes | 1 of 1 |  |
| no_medical_advice | 1 of 1 |  |

## Reply latency

From the caller's last word to the bot's first, as measured by the bot itself. The harness runs on the same machine, so these are a little slower than a real call.

| Replies | p50 | p95 | Slowest |
|---|---|---|---|
| 7 | 3337 ms | 4639 ms | 4639 ms |

| Stage | Samples | Median | p95 |
|---|---|---|---|
| Speech-to-text, per segment | 32 | 516 ms | 865 ms |
| LLM, time to first token | 45 | 620 ms | 684 ms |
| Voice, time to first audio | 22 | 1098 ms | 1435 ms |

## LLM cost

96,978 input and 3,597 output tokens across all calls, about $0.0115 in total at Haiku 5.5 list price.
That is $0.0009 per call, or $0.0015 per minute of call. Speech-to-text and the voice run locally, so they add hardware cost but no per-minute fee.
