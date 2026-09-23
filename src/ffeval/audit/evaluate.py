"""Score an auditor against the labelled claims.

    uv run python -m ffeval.audit.evaluate --baseline

Never read accuracy alone: at an 8% bad rate, always saying "supported" scores 92%.
That floor is printed next to every result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .auditor import audit_claims, baseline_verdict, looks_checkable
from .packet import FactsPacket
from .verdicts import UNFAITHFUL, Verdict

CASES_DIR = Path("eval/cases")
OUT_DIR = Path("eval/out")


@dataclass(frozen=True)
class LabelledClaim:
    """One sentence plus the verdict it was given. `source` = who labelled it."""

    id: str
    text: str
    label: Verdict
    category: str
    subtype: str | None
    evidence_ids: tuple[str, ...]
    source: str


def load_cases(path: Path) -> tuple[FactsPacket, list[LabelledClaim], int]:
    """Read one case file. Returns (packet, claims, rules_version)."""
    data = json.loads(path.read_text())
    packet = FactsPacket.load(data["packet"])
    claims, skipped = [], 0
    for c in data["claims"]:
        if not c.get("label"):
            skipped += 1
            continue
        claims.append(
            LabelledClaim(
                id=c["id"],
                text=c["text"],
                label=Verdict(c["label"]),          # raises on an unknown string
                category=c.get("category", ""),
                subtype=c.get("subtype"),
                evidence_ids=tuple(c.get("evidence_ids") or ()),
                source=c.get("source", ""),
            )
        )
    if skipped:
        print(f"  note: skipped {skipped} unlabelled claims in {path.name}")
    return packet, claims, data.get("rules_version", 0)


@dataclass(frozen=True)
class Scores:
    """confusion[(true, predicted)] = count."""

    n: int
    confusion: dict[tuple[Verdict, Verdict], int]

    def _rows(self, trues: set[Verdict]) -> int:
        """Total labelled with any verdict in `trues`."""
        return sum(v for (t, _), v in self.confusion.items() if t in trues)

    def _cells(self, trues: set[Verdict], preds: set[Verdict]) -> int:
        return sum(
            v for (t, p), v in self.confusion.items() if t in trues and p in preds
        )

    @property
    def recall_unfaithful(self) -> float:
        """Of the bad claims, how many were flagged as bad. Wrong flavour still counts."""
        total = self._rows(set(UNFAITHFUL))
        return self._cells(set(UNFAITHFUL), set(UNFAITHFUL)) / total if total else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """Of the good claims, how many were flagged anyway."""
        total = self._rows({Verdict.SUPPORTED})
        return self._cells({Verdict.SUPPORTED}, set(UNFAITHFUL)) / total if total else 0.0

    @property
    def accuracy(self) -> float:
        """Exact agreement."""
        hits = sum(v for (t, p), v in self.confusion.items() if t is p)
        return hits / self.n if self.n else 0.0


def score(truth: list[LabelledClaim], predicted: list[Verdict]) -> Scores:
    """Build the confusion matrix."""
    assert len(truth) == len(predicted), "a verdict went missing"
    conf = {(t, p): 0 for t in Verdict for p in Verdict}   # all 16 cells exist
    for c, p in zip(truth, predicted):
        conf[(c.label, p)] += 1
    return Scores(n=len(truth), confusion=conf)


def always_supported_baseline(truth: list[LabelledClaim]) -> Scores:
    """The degenerate auditor: everything is supported. The honest floor."""
    return score(truth, [Verdict.SUPPORTED] * len(truth))


def checkable_rate(truth: list[LabelledClaim]) -> float:
    """Share of claims carrying something falsifiable. A guard rail on the writer."""
    return sum(looks_checkable(c.text) for c in truth) / len(truth) if truth else 0.0


def print_report(
    scores: Scores,
    floor: Scores,
    truth: list[LabelledClaim],
    predicted: list[Verdict],
    rules_version: int,
    regex: Scores | None = None,
    label: str = "this auditor",
) -> None:
    src = Counter(c.source or "unlabelled-source" for c in truth)
    print(f"\n{scores.n} claims   rules_version {rules_version}")
    print("labelled by: " + ", ".join(f"{k} ({v})" for k, v in src.most_common()))

    order = list(Verdict)
    w = 15
    print("\nCONFUSION MATRIX   rows = label, cols = auditor")
    print(" " * w + "".join(f"{p.value:>{w}}" for p in order))
    for t in order:
        row = "".join(f"{scores.confusion[(t, p)]:>{w}}" for p in order)
        print(f"{t.value:<{w}}{row}")

    mid = f"{'regex baseline':>16}" if regex else ""
    print(f"\n{'':<24}{label:>14}{mid}{'always-supported':>18}")
    print("-" * (56 + (16 if regex else 0)))
    for name, a, r, b in (
        ("recall on unfaithful", scores.recall_unfaithful,
         regex.recall_unfaithful if regex else None, floor.recall_unfaithful),
        ("false alarms", scores.false_alarm_rate,
         regex.false_alarm_rate if regex else None, floor.false_alarm_rate),
        ("exact agreement", scores.accuracy,
         regex.accuracy if regex else None, floor.accuracy),
    ):
        mid = f"{r:>16.0%}" if r is not None else ""
        print(f"{name:<24}{a:>13.0%}{mid}{b:>18.0%}")

    print(f"\ncheckable claims: {checkable_rate(truth):.0%}")

    for cat in sorted({c.category for c in truth if c.category}):
        sub = [(c, p) for c, p in zip(truth, predicted) if c.category == cat]
        s = score([c for c, _ in sub], [p for _, p in sub])
        print(f"  {cat:<14} n={s.n:<4} recall {s.recall_unfaithful:.0%}"
              f"   exact {s.accuracy:.0%}")

    misses = [
        (c, p) for c, p in zip(truth, predicted)
        if c.label in UNFAITHFUL and p not in UNFAITHFUL
    ]
    if misses:
        print(f"\nMISSED BAD CLAIMS ({len(misses)}):")
        for c, p in misses:
            print(f"  {c.id}  label={c.label.value:<14} said={p.value:<14} {c.text[:46]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="the deterministic checker")
    ap.add_argument("--model", help="model id for the LLM auditor, e.g. gemini-3.6-flash")
    ap.add_argument("--cases", default=str(CASES_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    if not (args.baseline or args.model):
        sys.exit("pass --baseline or --model NAME")

    files = sorted(Path(args.cases).glob("*.json"))
    if not files:
        sys.exit(f"no case files in {args.cases}")

    all_truth: list[LabelledClaim] = []
    all_pred: list[Verdict] = []
    all_regex: list[Verdict] = []
    rows, version = [], 0
    for f in files:
        packet, claims, version = load_cases(f)
        regex = [baseline_verdict(packet, c.text) for c in claims]
        if args.model:
            # One call per packet, not per claim: the packet is most of the tokens.
            main_v = list(audit_claims(packet, [c.text for c in claims], args.model).verdicts)
        else:
            main_v = regex
        for c, v, rv in zip(claims, main_v, regex):
            all_truth.append(c)
            all_pred.append(v.verdict)
            all_regex.append(rv.verdict)
            rows.append({"case_file": f.name, "id": c.id, "text": c.text,
                         "label": c.label.value, "predicted": v.verdict.value,
                         "reason": v.reason, "evidence_ids": list(v.evidence_ids),
                         "regex_predicted": rv.verdict.value, "source": c.source})

    s = score(all_truth, all_pred)
    floor = always_supported_baseline(all_truth)
    regex_scores = score(all_truth, all_regex) if args.model else None
    print_report(s, floor, all_truth, all_pred, version,
                 regex=regex_scores, label=args.model or "regex baseline")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # A cited id that does not exist in the packet is a fabricated citation - the
    # auditor pointing at evidence it never saw. Counted, because "supported with
    # evidence" is worthless if the evidence is invented.
    valid = set()
    for f in files:
        pk, _, _ = load_cases(f)
        valid |= {x.id for x in pk.facts} | {x.id for x in pk.news} | set(pk.numbers())
        # valid follows numbers() rather than restating it, so growing that list can
        # never turn a real id into a phantom "fabricated citation" again.
    fabricated = {r["id"]: [x for x in r.get("evidence_ids", []) if x not in valid]
                  for r in rows}
    fabricated = {k: v for k, v in fabricated.items() if v}
    cited = sum(1 for r in rows if r.get("evidence_ids"))
    print(f"\nfabricated evidence ids: {len(fabricated)} of {cited} cited claims")
    for k, v in fabricated.items():
        print(f"  {k}: {v}")

    name = f"results_{args.model.replace('/', '_')}.json" if args.model else "baseline_results.json"
    (out / name).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out / name}")

    if s.accuracy <= floor.accuracy:
        sys.exit("FAIL: no better than always saying supported")
    print("PASS: beats the always-supported floor")


if __name__ == "__main__":
    main()
