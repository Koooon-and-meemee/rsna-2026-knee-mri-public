"""Ask the operator a question over Telegram and wait for the answer in a file.

The Telegram bot is long-polled by myclaw's agentd, so this process cannot call
getUpdates - a second poller would steal the operator's own messages.

The first attempt at a route around that was to have the assistant relay answers
into a file. It cannot work, and the reason is worth keeping: this script sends
its question with sendMessage, so the question reaches the operator's phone
WITHOUT passing through agentd. The assistant therefore has no idea the question
exists, and a bare "Ok" arriving afterwards has no referent for it to relay.

What does work needs nothing from the assistant at all. agentd logs every
inbound message to myclaw/memory/recent.md as `[MM-DD HH:MM] User: <text>`, so
the operator answers in Telegram exactly as normal and this script reads the log.
artifacts/DECISIONS.txt is still honoured as a manual override for answering from
a terminal.

    uv run python src/ask.py "Submit the 6-member (pv-hold 0.8906)?" --timeout 7200

Do NOT wrap the call in a shell `timeout` shorter than --timeout: the wrapper
wins and the waiter dies while the question is already out. When that happens
the question still reached the operator, and the monitor watching recent.md will
catch the reply anyway - but nothing is blocking on it, so the caller must not
assume an answer is being waited for. Happened once, 24 August, Q2.

Prints the answer on stdout and exits 0. Exits 1 with nothing on stdout if the
timeout passes unanswered, so a caller can fall back to its default rather than
block a queue that is otherwise making progress.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "artifacts" / "QUESTIONS.md"
DECISIONS = ROOT / "artifacts" / "DECISIONS.txt"
RECENT = Path("/home/sangwon-park/myclaw/memory/recent.md")


def next_id() -> int:
    if not QUESTIONS.exists():
        return 1
    ids = [int(m) for m in re.findall(r"^## Q(\d+)", QUESTIONS.read_text(), re.M)]
    return max(ids, default=0) + 1


def answer_for(qid: int) -> str | None:
    """Explicit answer in DECISIONS.txt. Last one wins if answered twice."""
    if not DECISIONS.exists():
        return None
    found = None
    for line in DECISIONS.read_text().splitlines():
        m = re.match(rf"\s*Q{qid}\s*[:=]\s*(.+)", line, re.I)
        if m and m.group(1).strip():
            found = m.group(1).strip()
    return found


def user_lines(path: Path) -> list[str]:
    """Every `User:` line agentd has logged, oldest first."""
    if not path.exists():
        return []
    return re.findall(r"^\[[\d-]+ [\d:]+\] User: (.+)$", path.read_text(), re.M)


def telegram_reply(baseline: int) -> str | None:
    """The first operator message logged after the question went out.

    Matched by position rather than by content: the operator answers in his own
    words and should not have to quote a question number. Anything he types
    after the question is sent is treated as the answer to it, which is why only
    one question is ever left open at a time.
    """
    lines = user_lines(RECENT)
    return lines[baseline].strip() if len(lines) > baseline else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--context", default="", help="extra detail for the record, not sent")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds to wait (0 = fire and forget)")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    qid = next_id()
    baseline = len(user_lines(RECENT))   # anything after this point is the answer
    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    with QUESTIONS.open("a") as f:
        f.write(f"\n## Q{qid} — {datetime.now():%Y-%m-%d %H:%M}\n\n{a.question}\n")
        if a.context:
            f.write(f"\n{a.context}\n")

    msg = f"Q{qid}. {a.question}"
    subprocess.run(["uv", "run", "python", str(ROOT / "src" / "notify.py"), msg],
                   cwd=ROOT, check=False)

    if not a.timeout:
        print(f"Q{qid} sent, not waiting", file=sys.stderr)
        return 1

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        ans = answer_for(qid) or telegram_reply(baseline)
        if ans:
            with DECISIONS.open("a") as f:
                f.write(f"Q{qid}: {ans}\n")
            print(ans)
            return 0
        time.sleep(a.poll)
    print(f"Q{qid} unanswered after {a.timeout}s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
