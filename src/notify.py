"""Send a short message to the operator over Telegram.

Used when a run needs a decision that cannot wait for the next time someone
opens the terminal - a submission worth spending a slot on, a job that died in a
way the queue cannot recover from, a result that changes the plan.

Credentials are read at call time from the geofence-alert service's env file,
which lives outside this repository. Nothing secret is stored here, and the
token must never be copied into the repo, a log line, or a commit message.

    uv run python src/notify.py "m46blend12 gold 0.9241 (+0.0078) - submitting"
    echo "..." | uv run python src/notify.py -

Exits non-zero and prints why if the channel is unavailable, so a caller can
fall back to the log rather than silently believing the message was delivered.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ENV_FILE = Path("/home/sangwon-park/geofence-alert/.env")
PREFIX = "[rsna] "


def load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def main() -> int:
    text = sys.stdin.read() if sys.argv[1:2] == ["-"] else " ".join(sys.argv[1:])
    text = text.strip()
    if not text:
        print("nothing to send", file=sys.stderr)
        return 2

    env = {**load_env(ENV_FILE), **os.environ}
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(f"no telegram credentials in {ENV_FILE} or environment", file=sys.stderr)
        return 3

    # Telegram caps a message at 4096 characters and rejects the whole thing
    # rather than truncating, so trim here where the tail is the least useful part.
    body = (PREFIX + text)[:4000]
    data = urllib.parse.urlencode({"chat_id": chat, "text": body}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with urllib.request.urlopen(url, data=data, timeout=30) as r:
            ok = r.status == 200
    except Exception as e:
        # Never print the exception's URL - it carries the token.
        print(f"telegram send failed: {type(e).__name__}", file=sys.stderr)
        return 4
    print("sent" if ok else "telegram rejected the message", file=sys.stderr)
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
