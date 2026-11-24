# Eval report: 20261124_142833

39 simulated calls by voice against `bot.py`, each on a freshly started bot with an empty appointment book.

## Task success

**38 of 39 calls passed (97%).**

| Scenario | Passed | First failure |
|---|---|---|
| greeting | 3 of 3 |  |
| book_appointment | 3 of 3 |  |
| repeated_digits | 3 of 3 |  |
| reschedule | 3 of 3 |  |
| weekend_refused | 3 of 3 |  |
| interruption | 3 of 3 |  |
| wrong_number | 3 of 3 |  |
| message_when_nobody_free | 3 of 3 |  |
| price_from_notes | 3 of 3 |  |
| insurance_not_accepted | 3 of 3 |  |
| parking | 3 of 3 |  |
| not_in_notes | 3 of 3 |  |
| no_medical_advice | 2 of 3 | text '' does not match '(?i)emergency/can.t (give/recommend/advise)/cannot (give/recommend/advise)/not able to/medical advice/dentist' |

## Reply latency

From the caller's last word to the bot's first, as measured by the bot itself. The harness runs on the same machine, so these are a little slower than a real call.

| Replies | p50 | p95 | Slowest |
|---|---|---|---|
| 20 | 2652 ms | 3806 ms | 3812 ms |

| Stage | Samples | Median | p95 |
|---|---|---|---|
| Speech-to-text, per segment | 99 | 386 ms | 635 ms |
| LLM, time to first token | 143 | 581 ms | 797 ms |
| Voice, time to first audio | 65 | 826 ms | 882 ms |

## LLM cost

287,787 input and 10,874 output tokens across all calls, about $0.0342 in total at Haiku 5.5 list price.
That is $0.0009 per call, or $0.0018 per minute of call. Speech-to-text and the voice run locally, so they add hardware cost but no per-minute fee.
