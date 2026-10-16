# How answers reach the Kaggle session

**Nothing is required of the operator or of the assistant.** Answer the Telegram
question in your own words, as normal. This document exists to record why the
obvious design does not work.

## The route

    session  --sendMessage-->  operator's phone        "[rsna] Q7. ...?"
    operator --Telegram----->  agentd  --logs to-->    myclaw/memory/recent.md
    session  --reads--------->  recent.md              "[08-24 12:41] User: Ok"

`src/ask.py` counts the `User:` lines in recent.md before sending, then treats
the next one as the answer. It also honours an explicit `Q<n>: ...` line in
`artifacts/DECISIONS.txt`, which is the way to answer from a terminal instead of
a phone.

Only one question is left open at a time, because the match is positional - the
operator should never have to quote a question number back.

## Why the assistant cannot relay it

The first design had the assistant append answers to DECISIONS.txt. It cannot
work. This session sends its question with `sendMessage`, so the question
reaches the phone **without passing through agentd** - agentd only ever sees
inbound messages. The assistant therefore has no idea the question exists, and a
bare "Ok" arriving afterwards has no referent. Tested on 24 August: the operator
answered, the assistant replied "네, 확인했습니다" and wrote nothing, correctly.

Reading agentd's own log needs no cooperation from it, no second poller stealing
its updates, and no standing rule in its persona file.

## If it breaks

- `getUpdates ... 409 Conflict` in `myclaw/agentd.log` means two processes are
  polling the bot. The Kaggle session never polls; look elsewhere. Note the
  entries dated after the current time are stale - the machine's clock was
  skewed at boot on 21 August.
- If recent.md stops growing, agentd is down, and the fallback is a `Q<n>:` line
  in DECISIONS.txt written over SSH.
