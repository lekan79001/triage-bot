# Support Ticket Triage Bot

## How to run it

No dependencies needed for the default (mock) mode — it's pure standard library:

```bash
python triage.py
```

This reads `tickets.csv`, writes `output.csv`, prints any high-urgency alerts to the
console, and appends them to `alerts.log`.

To call a real LLM instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python triage.py --live
```

Other flags: `--input`, `--output`, `--alerts-log` if you want different filenames.

## Design

- `mock_llm_call()` and `real_llm_call()` have the identical signature `(subject, body) -> str`
  (a JSON string). `run()` picks one based on `--live`. That's the only thing that changes
  going from mock to live — parsing, validation, retries, and output writing don't know or
  care which one produced the text.
- The mock doesn't just return canned "correct" answers — it randomly simulates a timeout,
  a rate limit, and a malformed JSON payload on a fraction of calls, so the retry/fallback
  logic actually gets exercised on every run rather than only looking correct on paper.
- `parse_and_validate()` checks the response is valid JSON *and* that `category`/`urgency`
  are from the allowed enums and `summary` is a non-empty string — a response that parses as
  JSON but has `"urgency": "urgent!!"` is treated as malformed, not accepted.
- `classify_ticket()` retries up to 3 times: exponential backoff for timeouts/rate limits
  (these are load problems — waiting helps), immediate retry for malformed JSON (a content
  problem — waiting won't fix bad output, but a fresh sample might).
- If a ticket still has no valid result after retries, it doesn't crash the batch or get
  dropped — it gets a `fallback_result()` flagged `category=other`, `urgency=medium`, with
  `[NEEDS MANUAL REVIEW]` in the summary, so a human sees it in `output.csv` instead of it
  silently vanishing.

## The prompt

```
You are a support ticket triage assistant. Given a ticket's subject and body,
respond with ONLY a JSON object (no prose, no markdown fences) with exactly these three fields:

- "category": one of "billing", "bug", "feature_request", "question", "other"
- "urgency": one of "low", "medium", "high"
- "summary": a single sentence (max ~20 words) summarizing the ticket

Ticket subject: {subject}
Ticket body: {body}

JSON:
```

It spells out the exact enum values rather than describing them in prose, because that's
what actually gets checked in `parse_and_validate()` — the model has no room to invent a
`category: "complaint"` that isn't one of the five we handle downstream. Explicitly saying
"no prose, no markdown fences" cuts down on the most common failure mode with real models:
wrapping JSON in a sentence or a ` ```json ` fence that breaks a naive `json.loads()`. I did
not add fence-stripping logic on top of this, since a clear instruction plus retry-on-failure
covers it without extra complexity — worth revisiting if it turns out to happen often in
practice.

## What I'd change for production (thousands of tickets/day)

- **Concurrency**: this processes tickets serially. At scale I'd batch calls with an async
  client and a bounded semaphore, so throughput doesn't sink linearly with ticket volume.
- **Cost**: cache or short-circuit near-duplicate tickets (e.g. the same user hitting "submit"
  three times), and consider a cheaper/smaller model for the classification step, reserving a
  larger model only for ambiguous cases.
- **Retries and rate limits**: replace the manual backoff loop with a proper retry library
  (e.g. `tenacity`) and respect `Retry-After` headers instead of guessing backoff intervals.
- **Monitoring**: track the fallback rate (how many tickets hit `[NEEDS MANUAL REVIEW]`) as a
  metric — a spike means either the prompt broke or the provider is degraded, and someone
  should get paged before a backlog builds up quietly.
- **Alerting**: swap `log_alert()`'s print/file-write for a real Slack webhook or PagerDuty
  integration, with rate-limiting so a bad batch doesn't spam the channel.
- **Idempotency**: persist which ticket IDs have already been processed (a small DB or
  processed-IDs file) so a crashed run can resume without re-billing already-classified
  tickets.

## Where I used AI

I used Claude to help scaffold this script, particularly the retry/backoff structure and the
mock-failure-simulation approach. I reviewed and adjusted the classification heuristics in
`mock_llm_call()` myself (they're intentionally simple keyword matching — not meant to be
accurate, just meant to produce plausible, schema-valid output for demonstrating the
pipeline), and manually ran the script multiple times to confirm the timeout, rate-limit, and
malformed-JSON paths actually trigger and recover rather than just looking correct on paper.

## Assumptions made

- No API key was available for this exercise, so the default path is `mock_llm_call()` per
  the assignment's instructions. The code is structured so `--live` is a one-flag switch to
  a real Anthropic call.
- Output format: CSV (matching the input format), with `category`, `urgency`, `summary`
  appended as the last three columns.
- "High urgency" tickets trigger the alert; I didn't add alerting for repeated failures,
  though that'd be a natural next step (see Monitoring above).
