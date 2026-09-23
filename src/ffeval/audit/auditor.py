"""Rule on each sentence of writer output against a facts packet.

Splitting is deterministic and happens in Python, never in the model. If the model chose
the units, they would change between runs, no two runs would be comparable, and the hand
labels - which are attached to specific sentences - would stop matching anything.

Two auditors live here, and the cheap one comes first:

  baseline_verdict()  no model at all. Does every number in the sentence appear in the
                      packet? Catches fabricated digits, misses everything about meaning.
                      This is the number the LLM auditor has to beat, and it is free.

  audit_claims()      the LLM auditor: one call judges a whole list of sentences.
                      Measured at 93% recall on unfaithful claims vs the baseline's 33%
                      - see docs/FINDINGS.md section 9.

Both judge sentences ONCE. The rewrite loop - flag, send back to the writer, re-judge,
budget of two, then delete the sentence - is not built, and cannot be until a writer
exists to rewrite anything.

Then a second layer, which is not an auditor at all:

  unsourced_numbers() the auditors ask "is this faithful?"; this asks "may it ship?"
                      A figure sourced only to a news snippet is faithful AND refused.
                      Shipping needs both layers, and keeping them apart is what makes
                      a bad score attributable to one of them.

Both layers read digits through _match_numbers, so they can never disagree about how a
number is read. They deliberately consult different lists of numbers, because "does the
packet confirm this?" and "does this rest on a structured field?" are not the same
question - see FactsPacket.numbers() vs sourced_numbers().

Numeric rule is Fork 1(a) from eval/LABELLING_RULES.md: a stated number matches a packet
value if the packet value ROUNDS to it at the precision the sentence used. "25" matches
25.29; "25.3" matches 25.29; "26" does not.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .packet import NUMBER as _NUMBER
from .packet import FactsPacket
from .verdicts import AuditResult, ClaimVerdict, Verdict

# Bumped whenever the LLM prompt text changes. Stored on every AuditResult so an old
# result file is never silently compared against a newer prompt.
PROMPT_VERSION = 2   # 2: one prompt per ROSTER, evidence ids namespaced per player

# Abbreviations whose full stop is not a sentence end. Only ones followed by a capital
# letter matter - "No. 3" already survives, because a digit is not a capital.
_ABBREVIATIONS = (
    "vs", "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr", "St", "Ave", "Inc", "Co",
    "approx", "etc", "e.g", "i.e", "No", "Nos", "Fig", "Sept", "Dec", "Jan",
)
_GUARD = "\x00"  # placeholder that cannot occur in real text

# A sentence ends at .!? only when followed by whitespace and something that starts a new
# sentence, or by the end of the string. This is what protects decimals for free: in
# "23.97" the dot is followed by a digit, not whitespace, so it never matches.
_SENTENCE_END = re.compile(r"""[.!?]+(?=\s+["'(\[]?[A-Z]|\s*$)""")

# Words that make a sentence checkable even with no digits in it.
_COMPARATIVE = re.compile(
    r"\b(most|least|best|worst|highest|lowest|more|less|fewer|better|worse|"
    r"stingiest|toughest|easiest|top|bottom|first|second|third|"
    r"said|told|reported|announced|according)\b",
    re.IGNORECASE,
)


def split_claims(text: str) -> list[str]:
    """Writer prose -> one string per sentence.

    Deliberately dull. Every claim id in eval/cases/ is anchored to this function's
    output, so changing it invalidates the labels.

    Newlines split first, so a bulleted list counts as one sentence per bullet even
    without full stops.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Hide abbreviation dots so the splitter cannot cut on them, then restore.
        hidden = line
        for abbr in _ABBREVIATIONS:
            hidden = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}{_GUARD}", hidden)

        start = 0
        for m in _SENTENCE_END.finditer(hidden):
            piece = hidden[start : m.end()].strip()
            if piece:
                out.append(piece.replace(_GUARD, "."))
            start = m.end()
        tail = hidden[start:].strip()
        if tail:
            out.append(tail.replace(_GUARD, "."))
    return out


def looks_checkable(claim: str) -> bool:
    """Does this sentence contain something that could be confirmed or refuted?

    A guard rail on the writer, not a verdict. An auditor creates pressure toward vague
    prose, because vagueness is unfalsifiable and so never flagged. A run at 0%
    unfaithful where nothing is checkable is a failure, and only this rate says so.
    """
    return bool(_NUMBER.search(claim) or _COMPARATIVE.search(claim))


def _rounds_to(stated: str, packet_value: float) -> bool:
    """Fork 1(a): does packet_value round to `stated` at the precision `stated` uses?"""
    decimals = len(stated.split(".")[1]) if "." in stated else 0
    try:
        return round(packet_value, decimals) == round(float(stated), decimals)
    except ValueError:
        return False


def _match_numbers(claim: str, numbers: dict[str, float]) -> tuple[list[str], list[str]]:
    """Every number in `claim`, split into (ids that back it, numbers nothing backs).

    Both layers share this so they can never disagree about how a digit is read. They
    pass DIFFERENT lists, because they disagree about which digits should count - see
    FactsPacket.numbers() vs sourced_numbers().
    """
    matched: list[str] = []
    unmatched: list[str] = []
    for s in _NUMBER.findall(claim):
        hits = [fid for fid, val in numbers.items() if _rounds_to(s, val)]
        if hits:
            matched.extend(hits)
        else:
            unmatched.append(s)
    return list(dict.fromkeys(matched)), unmatched


def baseline_verdict(packet: FactsPacket, claim: str) -> ClaimVerdict:
    """The free, no-model auditor. Beat this or the LLM adds nothing.

    Known blind spots, all deliberate - they are the argument for the LLM:
      - cannot produce NOT_IN_PACKET at all
      - cannot tell subject from object, so a right-number-wrong-player claim passes
      - small integers match promiscuously. A rank of 2 in the packet means any sentence
        containing "2" looks supported. Left in rather than patched, so the confusion
        matrix shows the cost instead of hiding it.
    """
    matched, unmatched = _match_numbers(claim, packet.numbers())
    if not matched and not unmatched:
        return ClaimVerdict(
            claim=claim,
            verdict=Verdict.NOT_A_CLAIM,
            evidence_ids=(),
            reason="no numbers in the sentence; this checker only reads numbers",
        )
    if unmatched:
        return ClaimVerdict(
            claim=claim,
            verdict=Verdict.CONTRADICTED,
            evidence_ids=(),
            reason=f"no packet fact rounds to {', '.join(unmatched)}",
        )
    return ClaimVerdict(
        claim=claim,
        verdict=Verdict.SUPPORTED,
        evidence_ids=tuple(matched),
        reason=f"every number matches a packet fact: {', '.join(matched)}",
    )


# ------------------------------------------------------- layer 2: the numeric-source gate


def unsourced_numbers(packet: FactsPacket, claim: str) -> list[str]:
    """The numbers in `claim` that rest on nothing structured. Empty means it may ship.

    A policy gate, not a verdict, and it disagrees with the auditor on purpose. "A beat
    writer noted a 34 percent pressure rate" is SUPPORTED - news.03 says exactly that -
    and still comes back as ["34"], because a figure sourced only to prose may not go out
    as the tool's own number. Folding this into the auditor's prompt instead would leave a
    bad score with two possible causes and nothing to tune against.

    What the caller does with a refused sentence - cut it, or send it back for a rewrite -
    is not decided here. There is no writer yet to rewrite anything.

    ponytail: digits only, so "over the last two games" is invisible to it. That hole
    closes when numeric prose is templated; a word-number parser now is work that gets
    deleted then.
    """
    return _match_numbers(claim, packet.sourced_numbers())[1]


# ------------------------------------------------------------------ the LLM auditor

CACHE_DIR = Path(".cache/model")


class QuotaExhausted(RuntimeError):
    """The daily free-tier allowance is gone. Distinct from every other failure.

    A 500 or a 503 is worth retrying; this is not - the free tier grants 20 requests per
    day per model, and no amount of backoff conjures the 21st. The caller's job is to ship
    what does not need a model rather than to try again.
    """

# The rules the model is held to. Kept as one string so the prompt and
# eval/LABELLING_RULES.md can be diffed by eye. Forks 1a, 2a, 3b, 4a.
_RULES = """You are auditing sentences against an evidence packet.

Your ONLY question per sentence: does it follow from the packet below?
NOT whether it is true in the real world. A sentence can be perfectly true and still
fail, because the packet does not contain it.

Answer with exactly one of:
  supported      - the packet contains this, and you can name the fact or news id
  contradicted   - the packet says something incompatible with this
  not_in_packet  - the packet neither confirms nor denies it, including true things it omits
  not_a_claim    - there is no factual assertion to check (a recommendation, a hedge)

Rules:
1. NUMBERS: a stated number is supported if a packet value rounds to it at the precision
   the sentence used. "25" matches 25.29. "25.3" matches 25.29. "26" does not.
2. DATES: the packet header names one season and week. A bare present-tense claim is about
   THAT week. If a news item from an earlier season says otherwise, the current structured
   fact wins and the sentence is contradicted.
3. ATTRIBUTION: a news item supports that SOMEONE SAID something, not the thing itself.
   "The coach said he expects a normal week" is supported. "He is expected to have a normal
   week", stated bare, is not_in_packet.
4. TWO CLAIMS IN ONE SENTENCE: give the worse verdict.
   contradicted > not_in_packet > supported > not_a_claim.
5. Every "supported" needs at least one id in evidence_ids. If you cannot name the
   evidence, it is not supported.
6. Each sentence is tagged with the packet it belongs to, like [p1]. Judge it against THAT
   packet only, and cite only ids from it. A p1 sentence may never rest on p2 evidence.
"""

_OUTPUT_FORMAT = """Reply with ONLY a JSON array, no prose and no code fence. One object per
sentence, in order:

[{"n": 1, "verdict": "supported", "evidence_ids": ["form.avg_ppr_l2"], "reason": "..."}]

Include every sentence exactly once. reason is one short sentence."""


def build_prompt(items: list[tuple[FactsPacket, list[str]]]) -> str:
    """One prompt for a whole roster: rules, every packet, every sentence, output format.

    A roster in one call rather than one call per player, because the free tier grants 20
    requests a day per model and a twelve-player roster was spending all of them. The
    writer has always worked this way; the auditor was the odd one out.

    Every packet's ids are namespaced with its tag, so a p1 sentence has no way to name a
    p2 fact. That is prevention rather than detection - the same reason the packet builder
    reads an allowlist of columns instead of blocking the bad ones.
    """
    blocks, numbered, n = [], [], 0
    for i, (packet, claims) in enumerate(items, start=1):
        tag = f"p{i}"
        blocks.append(
            f"--- EVIDENCE PACKET {tag} ({packet.player}) ---\n{packet.render(tag)}")
        for claim in claims:
            n += 1
            numbered.append(f"{n}. [{tag}] {claim}")
    return (
        f"{_RULES}\n"
        + "\n\n".join(blocks)
        + f"\n\n--- SENTENCES TO JUDGE ({n}) ---\n"
        + "\n".join(numbered)
        + f"\n\n{_OUTPUT_FORMAT}\n"
    )


def call_model(prompt: str, model: str, cache_dir: Path | str = CACHE_DIR) -> str:
    """One model call, cached on disk by (model, prompt).

    The cache is not an optimisation. Prompt iteration re-runs the same claims dozens of
    times; without it every tweak costs quota and no run is reproducible.
    """
    cache = Path(cache_dir)
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()[:16]
    hit = cache / f"{key}.txt"
    if hit.exists():
        return hit.read_text()

    from dotenv import load_dotenv          # imported here so tests never need a key
    from google import genai
    from google.genai import types

    load_dotenv()
    if model.startswith("groq/"):
        text = _generate_groq(model.removeprefix("groq/"), prompt)
    else:
        client = genai.Client()             # reads GEMINI_API_KEY
        text = _generate(client, types, model, prompt).text or ""
    cache.mkdir(parents=True, exist_ok=True)
    hit.write_text(text)
    return text


def _generate(client, types, model: str, prompt: str):
    """The call itself, with quota exhaustion given its own exception.

    A 429 is not a transient failure worth retrying - the daily allowance does not come
    back in sixty seconds. Naming it lets the pipeline ship the half of its output that
    needs no model at all.
    """
    from google.genai import errors

    config = types.GenerateContentConfig(
        temperature=0,
        # We pass no tools, so the SDK's function-calling setup is dead weight and warns
        # on every call. Off.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # Retries are OFF unless asked for - retry_options=None means "never retry" in
        # the SDK, which is not what the name suggests. Measured: gemma-4-31b-it failed 4
        # of 10 identical calls with 500 and 503, both transient and both already in the
        # SDK's retry list. An empty HttpRetryOptions() takes its defaults: 5 attempts,
        # exponential backoff with jitter, capped at 60s. Nothing hand-rolled - the
        # dependency already does this correctly.
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
    )
    try:
        return client.models.generate_content(
            model=model, contents=prompt, config=config
        )
    except errors.ClientError as e:
        if getattr(e, "code", None) == 429:
            raise QuotaExhausted(str(e)[:300]) from e
        raise


def groq_client():
    """Groq speaks the OpenAI wire format, so the openai package is the client.

    max_retries=5: requests go out back to back, so the next one usually lands in a
    minute the last one already spent. Groq's 429 says how long to wait and the SDK
    waits it (up to 60s). A 429 that outlives five waits is a daily limit, not a minute.
    """
    import os

    import openai
    from dotenv import load_dotenv

    load_dotenv()
    return openai.OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.environ["GROQ_API_KEY"], max_retries=5)


def _generate_groq(model: str, prompt: str) -> str:
    """A 429 that outlives the client's retries is treated exactly like Gemini's: the
    allowance is gone, take the fallback route."""
    import openai

    client = groq_client()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            # ponytail: fixed ceiling. gpt-oss "thinks" before answering and the thinking
            # counts against this; ~3K was enough once and too little the next time.
            # Raise it if finish_reason=length comes back again.
            max_completion_tokens=16384,
            messages=[{"role": "user", "content": prompt}])
    except openai.RateLimitError as e:
        raise QuotaExhausted(str(e)[:300]) from e
    # Measured: one reply came back cut off at claim 24 of 30. call_model caches whatever
    # returns, so a cut-off reply would be replayed forever. Refuse it before it is stored.
    choice = resp.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"groq reply unfinished: finish_reason={choice.finish_reason}")
    return choice.message.content or ""


_PACKET_TAG = re.compile(r"^p\d+/")


def _untag(evidence_id: str) -> str:
    """p1/form.game_w01 -> form.game_w01.

    The tag is a prompting device that stops one player's sentence citing another's
    evidence; it is not part of the data model. Letting it escape would make
    evaluate.py's fabricated-citation check report every real citation as invented,
    since it validates against the packet's own unprefixed ids.
    """
    return _PACKET_TAG.sub("", evidence_id)


def strip_fence(raw: str) -> str:
    """Model text -> the JSON inside it, tolerating a ```json fence around it."""
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        body = body[4:] if body.lower().startswith("json") else body
    return body.strip()


def parse_response(raw: str, claims: list[str]) -> tuple[ClaimVerdict, ...]:
    """Model text -> verdicts. Raises rather than guessing.

    A missing or unparseable ruling is a real failure. Backfilling the majority class
    here is how a broken auditor comes out looking accurate.
    """
    try:
        rows = json.loads(strip_fence(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return JSON: {e}\n{raw[:300]}") from e

    by_n = {int(r["n"]): r for r in rows}
    missing = [i for i in range(1, len(claims) + 1) if i not in by_n]
    if missing:
        raise ValueError(f"model skipped sentences {missing} of {len(claims)}")

    out = []
    for i, claim in enumerate(claims, start=1):
        r = by_n[i]
        out.append(
            ClaimVerdict(
                claim=claim,                          # ours, not the model's echo
                verdict=Verdict(str(r["verdict"]).strip().lower()),   # raises if unknown
                evidence_ids=tuple(_untag(str(x)) for x in (r.get("evidence_ids") or ())),
                reason=str(r.get("reason", "")),
            )
        )
    return tuple(out)


# ponytail: characters, not tokens (~3.6 chars per token, measured on the eval prompt).
# Groq's free tier caps a MINUTE at 8,000 tokens and refuses any single request bigger
# than that; a 12-player roster asked for 13,296. 12,000 chars is about 3 players.
# Count real tokens if a chunk is ever refused again.
_GROQ_PROMPT_CHARS = 12_000


def _chunks(items: list, limit: int):
    """Pack players into prompts under `limit` characters, in order. A lone player who is
    over the limit on his own still goes, alone - there is nothing smaller to send."""
    chunk: list = []
    for it in items:
        if chunk and len(build_prompt(chunk + [it])) > limit:
            yield chunk
            chunk = []
        chunk.append(it)
    if chunk:
        yield chunk


def audit_roster(
    items: list[tuple[FactsPacket, list[str]]], model: str
) -> list[AuditResult]:
    """Judge a whole roster. Returns one result per packet, in order.

    One call on Gemini, whose limit is 20 requests a DAY. Several on Groq, whose limit is
    8,000 tokens a MINUTE - the two free tiers ration opposite things, so "one call per
    roster" is Gemini's rule, not a universal one.
    """
    if not model.startswith("groq/"):
        return _audit_call(items, model)
    return [r for c in _chunks(items, _GROQ_PROMPT_CHARS) for r in _audit_call(c, model)]


def _audit_call(
    items: list[tuple[FactsPacket, list[str]]], model: str
) -> list[AuditResult]:
    """One model call for these packets."""
    flat = [c for _, claims in items for c in claims]
    if not flat:
        return [AuditResult((), model, PROMPT_VERSION) for _ in items]

    verdicts = parse_response(call_model(build_prompt(items), model), flat)
    out, k = [], 0
    for _, claims in items:
        out.append(AuditResult(tuple(verdicts[k:k + len(claims)]), model, PROMPT_VERSION))
        k += len(claims)
    return out


def audit_claims(packet: FactsPacket, claims: list[str], model: str) -> AuditResult:
    """One packet. A roster of one, so there is only ever one prompt to maintain."""
    return audit_roster([(packet, claims)], model)[0]


def audit(packet: FactsPacket, text: str, model: str) -> AuditResult:
    """Judge writer prose: split it, then audit the sentences."""
    return audit_claims(packet, split_claims(text), model)
