"""Language detection and per-study data-quality flags for the reports.

Two things live here.

**Language.** The first pass at this used Latin-script stopword lists only, so
it dumped a third of the corpus into "unknown". That bucket turned out to be
mostly Greek (321) and Cyrillic (220) - 12.3% of the data in scripts the
detector could not see at all. Script detection runs first now, because a
non-Latin script is decisive and a stopword hit is not.

**Quality flags.** Three defects that corrupt labels rather than merely
weakening them, found by reading the teacher's complaints:

  bilateral   describes BOTH knees in one report while the images are of one
              knee, so findings from the contralateral side get attributed to
              this study. This is wrong-label noise, not missing-label noise,
              and it is the worst kind. The dataset marks the unambiguous cases
              itself with "[BILATERAL NOTE: two reports filed under this
              study]" (27 studies); the prose patterns catch the rest.
  dummy       placeholder/test reports with no content.
  prior_comp  describes change relative to a previous exam, so some findings
              refer to a study we do not have.
  duplicate   183 studies (4.2%) share their report text verbatim with another
              study - one Turkish template covers 37 knees. The report cannot
              tell those knees apart, so they all receive one identical label.

Nothing here drops rows. The flags exist so training can down-weight these
studies; deleting 2% of the data would cost more than the noise does.
"""

from __future__ import annotations

import re

import pandas as pd

from clean_report import clean_report

SCRIPTS = {
    "el": r"[Ͱ-Ͽἀ-῿]",   # Greek
    "cyr": r"[Ѐ-ӿ]",                # Bulgarian / Russian / Serbian
    "ar": r"[؀-ۿ]",
    "he": r"[֐-׿]",
    "th": r"[฀-๿]",
    "cjk": r"[一-鿿぀-ヿ가-힯]",
}

STOPWORDS = {
    "es": r"\b(hallazgos|menisco|rodilla|derrame|rotura|impresi[óo]n|sin|con)\b",
    "tr": r"\b(mevcuttur|izlenmi|sinyal|d[üu]zey|y[ıi]rt[ıi]k|eklem|kemik)\b",
    "de": r"\b(kein|nachweis|kniegelenk|meniskus|erguss|unauff[äa]llig)\b",
    "nl": r"\b(geen|meniscus|kniegewricht|vocht|rechts|links|afwijking)\b",
    "fr": r"\b(genou|m[ée]nisque|[ée]panchement|sans|avec|pas de)\b",
    "it": r"\b(ginocchio|menisco|versamento|della|nella|non si)\b",
    "pt": r"\b(joelho|menisco|derrame|sem|com|n[ãa]o)\b",
    "hr": r"\b(koljen|meniskus|izljev|nema|desno|lijevo)\b",
    "en": r"\b(the|there|is|no|and|with|meniscus|effusion|intact)\b",
}

# The dataset ships an explicit marker where two reports were filed under one
# StudyInstanceUID. It is decisive - those reports genuinely describe two knees
# - so it is matched separately from the prose patterns, which only suggest it.
BILATERAL_MARKER = re.compile(r"\[BILATERAL NOTE", re.IGNORECASE)
BILATERAL = re.compile(
    r"\b(both knees|bilateral|beide kniegelenke|ambas rodillas|her iki diz|"
    r"les deux genoux|entrambe le ginocchia|двете колен|двустранно)\b",
    re.IGNORECASE,
)
DUMMY = re.compile(r"\btest report\b|\bdummy\b|\bsample report\b|\btest study\b", re.IGNORECASE)
PRIOR = re.compile(
    r"\bcompared (?:with|to) (?:the )?prior\b|\bprevious (?:study|exam|mri)\b|\bvergleich\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    t = text or ""
    # script first - it is decisive where it fires
    for code, pattern in SCRIPTS.items():
        if len(re.findall(pattern, t)) >= 10:
            return code
    low = t.lower()
    scores = {k: len(re.findall(p, low)) for k, p in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 2 else "unk"


def quality_flags(text: str) -> dict:
    t = text if isinstance(text, str) else ""
    return {
        "bilateral": bool(BILATERAL_MARKER.search(t)) or bool(BILATERAL.search(t)),
        "dummy": bool(DUMMY.search(t)) or len(t.strip()) < 80,
        "prior_comparison": bool(PRIOR.search(t)),
    }


def sample_weight(flags: dict) -> float:
    """Loss weight for a study given its defects.

    Bilateral is penalised hardest because it produces confidently WRONG
    labels - the contralateral knee's findings look exactly like real evidence
    to the extractor. A dummy report produces no labels worth learning from.

    The duplicate term is 1/sqrt(n) over the number of studies sharing the
    text. Reasoning: those n knees get one identical label, so whatever error
    that label carries is perfectly correlated across all n - the group cannot
    supply n independent labels no matter how many images it has. 1/sqrt(n) is
    a compromise, not a derivation: 1/n would throw away image diversity the
    group does still provide, and 1.0 would let a 37-study template group
    outvote 37 individually-reported knees.
    """
    if flags["dummy"]:
        return 0.0
    w = 1.0
    if flags["bilateral"]:
        w *= 0.35
    if flags["prior_comparison"]:
        w *= 0.7
    w /= float(flags.get("duplicate_group", 1)) ** 0.5
    return w


def annotate(df: pd.DataFrame, report_col: str = "Report") -> pd.DataFrame:
    r = df[report_col].fillna("")
    out = df.copy()
    out["lang"] = r.map(detect_language)
    flags = r.map(quality_flags).apply(pd.Series)

    # Duplication is a property of the corpus, not of one string, so it is
    # computed here rather than in quality_flags. Matching is done on the
    # CLEANED text: two copies of one template that differ only in a stripped
    # [DATE] are the same report for our purposes. Very short texts are
    # excluded - two empty reports matching is the dummy flag's problem.
    key = r.map(clean_report)
    sizes = key.map(key[key.str.len() > 50].value_counts()).fillna(1)
    flags["duplicate_group"] = sizes.astype(int).to_numpy()

    out = pd.concat([out, flags], axis=1)
    out["sample_weight"] = flags.apply(sample_weight, axis=1)
    return out


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    tr = annotate(pd.read_csv(ROOT / "data" / "train.csv"))
    print("=== language mix (corrected) ===")
    vc = tr.lang.value_counts()
    for k, v in vc.items():
        print(f"  {k:5} {v:5d}  ({v / len(tr) * 100:4.1f}%)")
    print("\n=== quality flags ===")
    for c in ["bilateral", "dummy", "prior_comparison"]:
        print(f"  {c:18} {int(tr[c].sum()):5d}  ({tr[c].mean() * 100:4.1f}%)")
    dup = tr.duplicate_group > 1
    print(f"  {'duplicate report':18} {int(dup.sum()):5d}  ({dup.mean() * 100:4.1f}%)")
    print(f"  {'largest dup group':18} {int(tr.duplicate_group.max()):5d} studies share one text")
    print(f"\n  studies at weight 0.0 : {int((tr.sample_weight == 0).sum())}")
    print(f"  studies down-weighted : {int(((tr.sample_weight > 0) & (tr.sample_weight < 1)).sum())}")
    print(f"  mean sample weight    : {tr.sample_weight.mean():.3f}")
    tr[
        [
            "StudyInstanceUID",
            "lang",
            "bilateral",
            "dummy",
            "prior_comparison",
            "duplicate_group",
            "sample_weight",
        ]
    ].to_csv(
        ROOT / "artifacts" / "report_quality.csv", index=False
    )
    print("\nwrote artifacts/report_quality.csv")
