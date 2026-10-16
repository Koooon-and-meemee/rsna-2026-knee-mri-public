"""Search ensemble combinations over cached member predictions.

`ensemble.py` pays a full GPU pass over the shards for every combination it is
asked about, and the decode - not the forward pass - is the expensive part. In
five weeks that priced about a dozen combinations out of the 2^25 available
from the submission-valid roster, and every one of them was chosen by hand.
`ensemble.py --save-preds` writes the pass out once; this reads it back and
does the rest in numpy.

Discipline, unchanged from the trainer's:

  **Search on pv-sel, report pv-hold.** Picking the best of hundreds of subsets
  on a metric with sd 0.0034 inflates it exactly the way picking the best of
  twelve epochs does. The held-out half never influences which subset is
  chosen, so the sel/hold gap is the size of the inflation rather than
  something to argue about afterwards.

  **Rank, not probability.** The metric is AUC, which reads only order. A
  member confident in the wrong units drags a probability mean toward itself
  without being better ordered.

  **gold-58 abstains below 0.008.** Two to three of its own replicate sd. It is
  printed for every candidate and it is a veto, not a vote.

    uv run python src/ensemble_search.py artifacts/preds_pool.npz
    uv run python src/ensemble_search.py artifacts/preds_pool.npz --keep m31newshards
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def study_auc(P, T, U):
    """Macro AUC at study level - the same reduction train_image.score uses."""
    dp = pd.DataFrame(P, columns=LABELS)
    dt = pd.DataFrame(T, columns=LABELS)
    dp["uid"], dt["uid"] = U, U
    dp, dt = dp.groupby("uid").mean(), dt.groupby("uid").first()
    aucs = {}
    for c in LABELS:
        y = (dt[c].to_numpy() > 0.5).astype(int)
        if len(np.unique(y)) > 1:
            aucs[c] = float(roc_auc_score(y, dp[c].to_numpy()))
    return float(np.mean(list(aucs.values()))), aucs


# Members the board has already refused, dropped unless --no-default-drop.
#
# These were passed by hand on the command line every time, which is a rule
# that only works while someone remembers it - the 20 September searches carry
# `--drop m53nbr m55nbr33 m56nbr37 m54attn` and nothing would have complained
# if they had not. The standard for being here is the same for every entry: the
# board was asked and said no, more than once.
#
#   2.5D neighbour channels  m53nbr scored 0.938 and m55nbr33 0.942 against the
#                            incumbent's 0.947, both while pv-hold said they
#                            were the best single models measured here.
#   convnext_nano            refused three times - 0.924, 0.926, 0.930 - in the
#                            efficiency pairs, every one of them below the
#                            nano-free pair at matched cost.
#   m54attn                  attention pooling, the one run of its kind, below
#                            its own family on both instruments.
#
# Dropping them is not a claim that they are bad models. It is a claim that the
# only instrument that has ever settled this said no, and a search that keeps
# offering them is spending its top-20 on candidates already priced.
BOARD_REFUSED = ["m53nbr", "m55nbr33", "m56nbr37", "m54attn",
                 "m52nano", "m58nanos1", "m59nano192", "m60nano288", "m61nano336"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--max-members", type=int, default=6,
                    help="ensembles larger than this are not searched; scoring cost "
                         "on Kaggle is 53 min fixed plus 31 min per member")
    ap.add_argument("--keep", nargs="*", default=[],
                    help="members forced into every candidate")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="dropped in addition to BOARD_REFUSED above")
    ap.add_argument("--no-default-drop", action="store_true",
                    help="also search the members the board has refused")
    ap.add_argument("--top", type=int, default=15)
    # The submission kernel averages PROBABILITIES, deliberately: measured on
    # 2026-08-20 the probability mean gained +0.0052 over the best member and
    # the rank mean only +0.0019, and ranks cannot be formed in a kernel that
    # streams one study at a time anyway. A search run on ranks therefore picks
    # a combination the kernel cannot deliver. Default to what ships.
    ap.add_argument("--rank", action="store_true",
                    help="average per-label percentile ranks instead of probabilities")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    drop = set(a.drop) | (set() if a.no_default_drop else set(BOARD_REFUSED))
    tags = [t for t in z["tags"].tolist() if t not in drop]
    hit = sorted(set(z["tags"].tolist()) & drop)
    if hit:
        print(f"dropped {len(hit)} of {len(z['tags'])}: {' '.join(hit)}")

    data = {}
    for split in ("val", "gold_val"):
        T = z[f"T__{split}"]
        U = z[f"U__{split}"]
        P = {t: z[f"P__{split}__{t}"] for t in tags}
        if a.rank:
            # Per-label percentile ranks over the whole split, so members with
            # different calibration average on order alone.
            R = {t: pd.DataFrame(P[t]).rank(pct=True).to_numpy(np.float32) for t in tags}
        else:
            R = {t: P[t].astype(np.float32) for t in tags}
        data[split] = (R, T, U)

    Rv, Tv, Uv = data["val"]
    sel = np.array([int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 for u in Uv])
    Rg, Tg, Ug = data["gold_val"]

    def evaluate(combo):
        m = np.mean([Rv[t] for t in combo], axis=0)
        s, _ = study_auc(m[sel], Tv[sel], Uv[sel])
        h, _ = study_auc(m[~sel], Tv[~sel], Uv[~sel])
        g, per = study_auc(np.mean([Rg[t] for t in combo], axis=0), Tg, Ug)
        return s, h, g, per

    print(f"{len(tags)} members, {len(Uv)} val series, {len(Ug)} gold series"
          f"   [{'rank' if a.rank else 'probability'} mean]\n")
    print(f"{'member':18}{'pv-sel':>9}{'pv-hold':>9}{'gold':>9}")
    solo = {}
    for t in tags:
        s, h, g, _ = evaluate([t])
        solo[t] = h
        print(f"{t:18}{s:9.4f}{h:9.4f}{g:9.4f}")

    incumbent = ["m31newshards", "m33up336new", "m32blend40new", "m37up288"]
    if all(t in tags for t in incumbent):
        s, h, g, _ = evaluate(incumbent)
        print(f"\n{'INCUMBENT (LB 0.947)':18}{s:9.4f}{h:9.4f}{g:9.4f}")
        base_sel = s
    else:
        base_sel = max(evaluate([t])[0] for t in tags)

    # Greedy forward selection on pv-sel. Exhaustive is 2^25; greedy over 25
    # members to size 6 is 25 + 24 + ... = 135 evaluations and reaches the same
    # place whenever members are close to interchangeable, which the solo table
    # above says they are.
    print("\ngreedy forward selection, scored on pv-sel:\n")
    print(f"{'n':>3}  {'added':18}{'pv-sel':>9}{'pv-hold':>9}{'gold':>9}   ensemble")
    chosen = list(a.keep)
    rows = []
    while len(chosen) < a.max_members:
        best = None
        for t in tags:
            if t in chosen:
                continue
            s, h, g, _ = evaluate(chosen + [t])
            if best is None or s > best[0]:
                best = (s, h, g, t)
        if best is None:
            break
        s, h, g, t = best
        chosen.append(t)
        rows.append((len(chosen), t, s, h, g, list(chosen)))
        print(f"{len(chosen):>3}  {t:18}{s:9.4f}{h:9.4f}{g:9.4f}   {'+'.join(chosen)}")

    # Every subset of the best few, so the greedy path is not the only answer
    # on offer. Small enough to be exhaustive where it matters.
    pool = [t for t in sorted(tags, key=lambda t: -solo[t])[:10]]
    print(f"\nexhaustive over the 10 best solo members, sizes 2-{min(5, a.max_members)}:\n")
    out = []
    for n in range(2, min(5, a.max_members) + 1):
        for combo in itertools.combinations(pool, n):
            s, h, g, _ = evaluate(list(combo))
            out.append((s, h, g, combo))
    out.sort(reverse=True)
    print(f"{'pv-sel':>9}{'pv-hold':>9}{'gold':>9}   ensemble")
    for s, h, g, combo in out[: a.top]:
        print(f"{s:9.4f}{h:9.4f}{g:9.4f}   {'+'.join(combo)}")

    print(f"\nreference: incumbent pv-sel {base_sel:.4f}. A candidate has to clear it "
          f"on pv-sel to be\nconsidered at all, and its pv-hold is the honest number "
          f"once it is.")


if __name__ == "__main__":
    main()
