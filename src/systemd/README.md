# systemd user units

The GPU queue used to be hand-started with `nohup`. Between 13 and 19 August
2026 the machine rebooted five times, nothing restarted the runner, and the
card sat idle for six days while a training run that had been climbing died at
epoch 3 of 16. Six idle days cost more than any single experiment on the open
list was going to win, so the runner is a service now.

Lingering is already enabled for this user (`loginctl show-user $USER -p
Linger`), so these start at boot without anyone logging in.

    cp src/systemd/*.service src/systemd/*.timer ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now rsna-queue.service rsna-lb.timer

    systemctl --user status rsna-queue      # is the queue draining
    systemctl --user list-timers rsna-lb    # when the board is pulled next
    tail -f artifacts/logs/queue.log

`rsna-queue.service` drains `artifacts/queue.txt`, one job at a time.
`Restart=always` is deliberate even though an empty queue is not a failure:
restarting is what makes a line appended to the queue start on its own instead
of waiting for someone to notice.

`rsna-lb.timer` pulls the public leaderboard once a day at 09:30 into
`artifacts/lb/`, so board velocity is a series rather than a memory.

**The one gotcha:** systemd user services get a minimal PATH without
`~/.local/bin`, so `uv` is not on it and every job dies instantly with
`failed to run command 'uv'`. Both units set PATH explicitly. Cost one launch
to find on 19 August.
