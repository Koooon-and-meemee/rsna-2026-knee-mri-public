"""Snapshot the public leaderboard once a day, so board velocity is a series.

On 2026-08-19 the question "has the leaderboard stabilised?" could not be
answered, because the only numbers on file were two hand-copied lines in
HANDOFF.md from the 10th and the 13th. The board itself is a full CSV that
Kaggle hands out on request, and the interesting quantity is not our rank -
it is where the 10th-place line sits and how fast it moves, because that line
is what separates $5,000 from nothing.

One file per day under artifacts/lb/. Kept small (about 60KB each), so a run
to the deadline costs a few megabytes.

    uv run python src/lb_snapshot.py            # pull today, print the summary
    uv run python src/lb_snapshot.py --report   # print the series, pull nothing
"""

from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

COMP = "rsna-knee-abnormality-detection"
TEAM = "qkrtkddnjs"
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "lb"


def pull(day: str) -> Path:
    dest = OUT / f"{day}.csv"
    if dest.exists():
        return dest
    OUT.mkdir(parents=True, exist_ok=True)
    # The CLI writes a zip named after the competition; unpack in memory rather
    # than leaving the archive and a dated csv both lying around.
    subprocess.run(
        ["uv", "run", "kaggle", "competitions", "leaderboard", COMP, "-d", "-p", str(OUT)],
        check=True, capture_output=True,
    )
    zpath = OUT / f"{COMP}.zip"
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        dest.write_bytes(z.read(name))
    zpath.unlink()
    return dest


def summarise(path: Path) -> dict:
    d = pd.read_csv(path).sort_values("Score", ascending=False).reset_index(drop=True)
    us = d[d.TeamName == TEAM]
    return {
        "day": path.stem,
        "teams": len(d),
        "us": float(us.Score.iloc[0]) if len(us) else float("nan"),
        "rank": int(us.Rank.iloc[0]) if len(us) else -1,
        "1st": float(d.Score.iloc[0]),
        "3rd": float(d.Score.iloc[2]),
        "10th": float(d.Score.iloc[9]),
        "20th": float(d.Score.iloc[19]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the series, pull nothing")
    args = ap.parse_args()

    if not args.report:
        pull(date.today().isoformat())

    rows = [summarise(p) for p in sorted(OUT.glob("20*.csv"))]
    if not rows:
        print("no snapshots yet")
        return
    print(f"{'day':<12}{'teams':>7}{'us':>8}{'rank':>6}{'1st':>8}{'3rd':>8}{'10th':>8}{'20th':>8}")
    for r in rows:
        print(f"{r['day']:<12}{r['teams']:>7}{r['us']:>8.3f}{r['rank']:>6}"
              f"{r['1st']:>8.3f}{r['3rd']:>8.3f}{r['10th']:>8.3f}{r['20th']:>8.3f}")
    if len(rows) > 1:
        a, b = rows[0], rows[-1]
        print(f"\nsince {a['day']}: 10th-place line {a['10th']:.3f} -> {b['10th']:.3f} "
              f"({b['10th'] - a['10th']:+.3f}), our rank {a['rank']} -> {b['rank']}")


if __name__ == "__main__":
    main()
