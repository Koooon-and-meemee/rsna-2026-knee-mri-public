"""Repair de-identification damage in the reports before they reach a model.

The de-identification pass that produced this dataset replaced some numeric
tokens with the literal string "intact". That is uniquely unfortunate: "intact"
is the standard radiology word for "this structure is normal", so a corrupted
measurement reads as a negative finding.

    "Small osteochondral body, about intact9xintact4cm"   (was ~0.9x0.4cm)
    "A intact3 cm subchondral bone cyst is present"       (was ~1.3 cm)

87 of 4,407 reports (2.0%) contain it. Left alone it pushes those studies
toward false negatives on exactly the findings the sentence is describing.

Also normalises the placeholder tokens ([DATE] in 14.5% of reports, [REDACTED]
in 2.0%) so they do not read as content.
"""

from __future__ import annotations

import re

# "intact" glued to a digit is always corruption - the real word never appears
# that way. Drop the token and keep the number.
#
# No \b on the left: the corruption lands inside dimension strings, where the
# preceding character is itself a word character ("3xintact5x1.8 cm"), and a
# word boundary there does not match. The lookahead for a digit is what makes
# this safe - no real word ending in "intact" is followed by a digit.
_INTACT_NUM = re.compile(r"intact(?=\d)", re.IGNORECASE)

# A bare "intact" wedged between a number/'x' and a unit is the same bug with
# the digits fully eaten, e.g. "1 x intact cm".
_INTACT_DIM = re.compile(
    r"(?<=[\dx×]\s)intact(?=\s*(?:x|×|cm|mm)\b)", re.IGNORECASE
)

_REDACTED = re.compile(r"\[REDACTED\]", re.IGNORECASE)

# Two shapes of de-identification placeholder. The named list was written from
# the tokens we had seen; auditing the corpus turned up [TIME] (359), [YEAR]
# (14), [PROFESSION] and [AFFILIATION] that it did not cover, so the second
# pattern catches any bare ALL-CAPS bracketed token generically.
#
# It deliberately requires the brackets to hold nothing but the token: the one
# bracketed marker that carries real information, "[BILATERAL NOTE: two reports
# filed under this study]", has a colon and prose and must survive - the reader
# needs to know two knees are described.
# Case-insensitive for the known tokens, which appear in mixed case...
_PLACEHOLDER = re.compile(
    r"\[(?:DATE|NAME|ID|AGE|HOSPITAL|ADDRESS)[^\]]*\]", re.IGNORECASE
)
# ...but case-SENSITIVE for the generic catch-all. Lowercasing it would also
# swallow ordinary bracketed prose like "[see above]", which is content.
_PLACEHOLDER_CAPS = re.compile(r"\[[A-Z][A-Z0-9 _/-]{1,24}\]")
_WS = re.compile(r"[ \t]{2,}")


def clean_report(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = _INTACT_NUM.sub("", text)
    t = _INTACT_DIM.sub("?", t)
    # A redacted findings line carries no information; make that explicit
    # rather than letting the model read "[REDACTED]" as a token.
    t = _REDACTED.sub("(not available)", t)
    t = _PLACEHOLDER.sub("", t)
    t = _PLACEHOLDER_CAPS.sub("", t)
    return _WS.sub(" ", t).strip()


def corruption_flags(text: str) -> dict[str, bool]:
    """Per-report flags, so downstream code can down-weight damaged studies."""
    t = text if isinstance(text, str) else ""
    return {
        "had_intact_corruption": bool(_INTACT_NUM.search(t) or _INTACT_DIM.search(t)),
        "had_redaction": bool(_REDACTED.search(t)),
    }


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    before = tr.Report.fillna("")
    after = before.map(clean_report)
    changed = (before != after).sum()
    print(f"{changed}/{len(tr)} reports changed ({changed / len(tr) * 100:.1f}%)")
    for b, a in zip(before, after):
        if "intact" in b.lower() and b != a and re.search(r"intact\d", b, re.I):
            i = re.search(r"intact\d", b, re.I).start()
            print("\n  before:", b[max(0, i - 45) : i + 45].replace("\n", " "))
            j = max(0, i - 45)
            print("  after :", a[j : j + 90].replace("\n", " "))
            break
