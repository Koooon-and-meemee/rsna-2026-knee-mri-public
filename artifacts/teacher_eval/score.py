"""Score a candidate teacher's labels against the 58 real radiologist labels.

Pure standard library - no pandas, no sklearn - so it runs anywhere python3 does.

    python3 score.py my_labels.csv

my_labels.csv needs a header with `id` (0-57, matching gold58_reports.json) and
the twelve finding columns, each a probability in [0, 1]. Column order does not
matter; extra columns are ignored. Any finding you leave out is skipped and the
mean is taken over what remains, so a partial run still tells you something.

Baselines to beat, on these same 58 studies (mean AUC over the 12 findings):

    fable    0.9004      the labeller that produced the 1,000-study set
    student  0.8993      the LoRA distillation that labels the full corpus

Read the per-finding column, not just the mean. The teacher is already 0.99 on
ACL and 0.79 on Synovitis, so a candidate that ties on the mean but fixes
Synovitis is worth more than one that ties by being better at ACL.

58 studies puts roughly +/-0.06 of standard error on any single finding, so
treat individual rows as indicative and the ordering as the result.
"""
import csv, sys

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

FABLE = {'ACL': 0.996, 'MCL': 0.983, 'Medial Meniscus': 0.963, 'Lateral Meniscus': 0.89, 'Medial OA': 0.945, 'Lateral OA': 0.829, 'PF OA': 0.925, 'Effusion': 0.847, 'Synovitis': 0.751, "Baker's": 0.923, 'Contusion': 0.872, 'Fracture': 0.881}

STUDENT = {'ACL': 0.994, 'MCL': 0.972, 'Medial Meniscus': 0.952, 'Lateral Meniscus': 0.865, 'Medial OA': 0.919, 'Lateral OA': 0.818, 'PF OA': 0.911, 'Effusion': 0.861, 'Synovitis': 0.787, "Baker's": 0.911, 'Contusion': 0.876, 'Fracture': 0.925}


def auc(y, p):
    """Mann-Whitney U, with ties taking the average rank."""
    pairs = sorted(zip(p, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    s = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load(path, key=None):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    truth = load("gold58_truth.csv")
    pred = load(sys.argv[1])
    if len(pred) != len(truth):
        print(f"warning: {len(pred)} predictions against {len(truth)} studies")
    # gold58_truth.csv row i corresponds to report id i - verified against the
    # existing fable output, which scores 0.9004 this way and ~0.5 any other.
    by_id = {}
    for row in pred:
        try:
            by_id[int(float(row["id"]))] = row
        except (KeyError, ValueError):
            raise SystemExit("every row needs an integer `id` column (0-57)")

    print(f"{'finding':<18}{'pos':>5}{'yours':>8}{'fable':>8}{'student':>9}")
    got = []
    for f in FINDINGS:
        if f not in pred[0]:
            print(f"{f:<18}{'':>5}{'absent':>8}")
            continue
        y, p = [], []
        for i, t in enumerate(truth):
            if i not in by_id:
                continue
            y.append(int(float(t[f])))
            p.append(float(by_id[i][f]))
        a = auc(y, p)
        if a is None:
            continue
        got.append(a)
        print(f"{f:<18}{sum(y):>5}{a:>8.3f}{FABLE[f]:>8.3f}{STUDENT[f]:>9.3f}")

    if got:
        print(f"\nmean over {len(got)} findings: {sum(got)/len(got):.4f}"
              f"   (fable 0.9004, student 0.8993)")


if __name__ == "__main__":
    main()
