"""
triage.py — Support ticket triage bot.

Reads tickets.csv, classifies each ticket via an LLM into category/urgency/summary,
writes output.csv, and logs an alert for every high-urgency ticket.

Usage:
    python triage.py                    # mock LLM, no API key needed
    python triage.py --live             # real Anthropic API (needs ANTHROPIC_API_KEY, `pip install anthropic`)
    python triage.py --input other.csv --output out.csv --alerts-log alerts.log
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("triage")

VALID_CATEGORIES = {"billing", "bug", "feature_request", "question", "other"}
VALID_URGENCIES = {"low", "medium", "high"}

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5


@dataclass
class TriageResult:
    category: str
    urgency: str
    summary: str


class LLMError(Exception):
    """Base class for anything that goes wrong calling or parsing the LLM."""


class RateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class MalformedResponseError(LLMError):
    pass


# ---------------------------------------------------------------------------
# LLM call — swap mock_llm_call for real_llm_call via --live. Everything
# downstream (parsing, validation, retries) doesn't care which one ran.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a support ticket triage assistant. Given a ticket's subject and body, \
respond with ONLY a JSON object (no prose, no markdown fences) with exactly these three fields:

- "category": one of "billing", "bug", "feature_request", "question", "other"
- "urgency": one of "low", "medium", "high"
- "summary": a single sentence (max ~20 words) summarizing the ticket

Ticket subject: {subject}
Ticket body: {body}

JSON:"""


def mock_llm_call(subject: str, body: str) -> str:
    """
    Stand-in for a real LLM call. Returns a JSON string in the same shape a
    real provider would. It also randomly simulates the three failure modes
    the exercise asks for (timeout, rate limit, malformed JSON) so the
    retry/fallback logic actually gets exercised, not just written.
    """
    roll = random.random()
    if roll < 0.07:
        raise LLMTimeoutError("mock LLM call timed out")
    if roll < 0.14:
        raise RateLimitError("mock LLM call was rate-limited (429)")

    text = f"{subject} {body}".lower()

    if any(w in text for w in ["charge", "refund", "billing", "cancel", "subscription"]):
        category = "billing"
    elif any(w in text for w in ["crash", "freeze", "error", "bug", "missing", "disappear"]):
        category = "bug"
    elif any(w in text for w in ["would be great", "feature", "export", "nice-to-have"]):
        category = "feature_request"
    elif "?" in body or "curious" in text or "discount" in text:
        category = "question"
    else:
        category = "other"

    if any(w in text for w in ["urgent", "immediately", "today", "asap", "in an hour"]):
        urgency = "high"
    elif category in ("bug", "billing"):
        urgency = "medium"
    else:
        urgency = "low"

    summary = (body.strip().split(".")[0].strip() + ".")[:140]

    # Occasionally hand back a broken payload to prove the parser catches it.
    if roll < 0.16:
        return '{"category": "' + category + '", "urgency": ' + urgency  # malformed on purpose

    return json.dumps({"category": category, "urgency": urgency, "summary": summary})


def real_llm_call(subject: str, body: str) -> str:
    """
    Real call to the Anthropic API. Requires `pip install anthropic` and
    ANTHROPIC_API_KEY set in the environment. This is the one function you
    swap to go from mock to live.
    """
    import anthropic  # imported lazily so the mock path has zero hard dependencies

    client = anthropic.Anthropic()
    prompt = PROMPT_TEMPLATE.format(subject=subject, body=body)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APITimeoutError as e:
        raise LLMTimeoutError(str(e)) from e
    except anthropic.RateLimitError as e:
        raise RateLimitError(str(e)) from e
    except anthropic.APIError as e:
        raise LLMError(str(e)) from e

    return response.content[0].text


def parse_and_validate(raw_text: str) -> TriageResult:
    """Parse + validate the LLM's raw text against our schema, or raise MalformedResponseError."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise MalformedResponseError(f"response was not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise MalformedResponseError("response JSON was not an object")

    category = data.get("category")
    urgency = data.get("urgency")
    summary = data.get("summary")

    if category not in VALID_CATEGORIES:
        raise MalformedResponseError(f"invalid category: {category!r}")
    if urgency not in VALID_URGENCIES:
        raise MalformedResponseError(f"invalid urgency: {urgency!r}")
    if not isinstance(summary, str) or not summary.strip():
        raise MalformedResponseError("summary missing or empty")

    return TriageResult(category=category, urgency=urgency, summary=summary.strip())


def classify_ticket(subject: str, body: str, llm_call, max_retries: int = MAX_RETRIES) -> Optional[TriageResult]:
    """
    Call the LLM and validate its output, retrying on timeouts, rate limits,
    and malformed JSON with exponential backoff. Returns None if every
    attempt fails, so the caller can fall back instead of crashing the batch.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = llm_call(subject, body)
            return parse_and_validate(raw)
        except RateLimitError as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning("Rate limited (attempt %d/%d). Backing off %.1fs.", attempt, max_retries, wait)
            time.sleep(wait)
        except LLMTimeoutError as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning("Timeout (attempt %d/%d). Retrying in %.1fs.", attempt, max_retries, wait)
            time.sleep(wait)
        except MalformedResponseError as e:
            last_error = e
            logger.warning("Malformed LLM response (attempt %d/%d): %s", attempt, max_retries, e)
            # No backoff — this is a content problem, not a load problem.
        except LLMError as e:
            last_error = e
            logger.warning("LLM error (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(BASE_BACKOFF_SECONDS)

    logger.error("Giving up on ticket after %d attempts: %s", max_retries, last_error)
    return None


def fallback_result(error: str) -> TriageResult:
    """What a ticket gets when the LLM never produces a usable answer — flagged for human review, not dropped."""
    return TriageResult(category="other", urgency="medium", summary=f"[NEEDS MANUAL REVIEW] {error}")


def log_alert(ticket_id: str, subject: str, summary: str, alerts_log_path: str) -> None:
    message = (
        f"ALERT — Ticket #{ticket_id}: \"{subject}\"\n"
        f"    Summary: {summary}\n"
        f"    Channel: #support-urgent (simulated)\n"
    )
    print(message)
    with open(alerts_log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def run(input_path: str, output_path: str, alerts_log_path: str, live: bool) -> None:
    llm_call = real_llm_call if live else mock_llm_call
    if not live:
        logger.info("Running in MOCK mode (no API key needed). Pass --live to call a real LLM.")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tickets = list(reader)

    if not tickets:
        logger.warning("Input file %s had no rows. Nothing to do.", input_path)
        return

    results = []
    failures = 0

    for ticket in tickets:
        ticket_id = ticket["id"]
        subject = ticket["subject"]
        body = ticket["body"]

        result = classify_ticket(subject, body, llm_call)

        if result is None:
            failures += 1
            result = fallback_result("LLM failed after retries")

        results.append({**ticket, **asdict(result)})

        if result.urgency == "high":
            log_alert(ticket_id, subject, result.summary, alerts_log_path)

    fieldnames = list(tickets[0].keys()) + ["category", "urgency", "summary"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(
        "Done. %d tickets processed, %d needed manual-review fallback. Output: %s",
        len(results), failures, output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Triage support tickets with an LLM.")
    parser.add_argument("--input", default="tickets.csv", help="Path to input CSV (default: tickets.csv)")
    parser.add_argument("--output", default="output.csv", help="Path to output CSV (default: output.csv)")
    parser.add_argument("--alerts-log", default="alerts.log", help="Path to alerts log (default: alerts.log)")
    parser.add_argument("--live", action="store_true", help="Call a real LLM instead of the mock")
    args = parser.parse_args()

    try:
        run(args.input, args.output, args.alerts_log, args.live)
    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
