#!/usr/bin/env bash
# Keep the GPU busy without waiting for a human or a scheduler.
#
# The 5090 is on loan for about two days, so an idle minute is a minute of the
# loan spent on nothing. A 30-minute check-in can only bound idle time at 30
# minutes; this bounds it at POLL seconds, because it starts the next run the
# moment a slot frees rather than the next time someone looks.
#
# Reads artifacts/queue.txt, one shell command per line, '#' comments ignored.
# The file is re-read every poll, so experiments can be appended to the backlog
# while the runner is going - which is the point, since what is worth running
# next depends on what the last run showed.
#
#   artifacts/queue.txt        pending, top line runs next
#   artifacts/queue.started    what has been launched, with a timestamp
#
# A launch is refused unless the GPU has FREE_MIN MiB spare. Two runs that OOM
# each other cost more than one run that waited: the failure lands mid-epoch,
# after the cache is warm and the time is already spent.
#
#   nohup bash src/gpu_queue.sh > $CLAUDE_JOB_DIR/tmp/queue.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
QUEUE=artifacts/queue.txt
STARTED=artifacts/queue.started
# 2026-08-19: logs used to land in $CLAUDE_JOB_DIR/tmp, which is deleted with
# the job. m32blend40new died at epoch 3 during the 13-19 Aug gap and its log
# was already gone by the time anyone looked, so the cause is permanently
# unknown. Default somewhere that outlives both the job and a reboot.
LOGDIR="${RSNA_LOGDIR:-$PWD/artifacts/logs}"
# One at a time, deliberately. A single training run already holds the GPU at
# 90-98% utilisation, so concurrency buys almost nothing: convnext_tiny takes
# 2.3 min/epoch alone and 6.6 min/epoch with three runs sharing the card, which
# is 3 x (2.3/6.6) = 1.05x total throughput. The 5% is paid for with 3x the
# latency on every result - and results are what decide the next experiment, so
# latency is the expensive axis, not throughput.
MAX_JOBS=${MAX_JOBS:-1}
FREE_MIN=${FREE_MIN:-9000}      # MiB; convnext_small at batch 8 needs ~8GB
POLL=${POLL:-60}

mkdir -p "$LOGDIR"
touch "$QUEUE" "$STARTED"

# Exactly one runner, enforced rather than remembered. Restarting this by hand
# after a stall has twice left the previous one alive, and two runners do not
# take turns: each sees a free slot during the other's `uv run` startup window,
# so they launch two lines seconds apart and both land on the card. That is how
# m89dinofrz and m90dino224 ended up sharing the GPU on 19 Sep, which cost
# m89 an hour of wall time it did not need to spend.
exec 9>"$LOGDIR/.queue.lock"
if ! flock -n 9; then
  echo "$(date '+%F %T') another runner already holds $LOGDIR/.queue.lock - exiting"
  exit 0
fi

# Match the interpreter, not the string. `pgrep -f src/train_image.py` also
# matches the `uv run` wrapper, this script's own launch line, and any shell
# that merely mentions the path - it reported 7 for 3 real runs, which pinned
# running() above MAX_JOBS forever and would have left the GPU idle the moment
# the current batch finished.
# src/train_ so that train_study.py counts too, not just train_image.py - a
# pattern naming one trainer makes every other trainer invisible to the slot
# check, which is how three runs ended up sharing the card.
#
# 2026-08-13: the same bug, one script wider. oof_predict.py isn't under
# src/train_, so it was invisible to this check - the queue saw "0 running"
# for the 140 minutes oof_predict held the GPU training 5 folds, and launched
# three more entries on top of it. blend_targets and m29blend40 failed cleanly
# on a missing input file (harmless, just wrong order); m23swin collided on
# VRAM and died with a real CUDA OOM. Any script this queue launches has to be
# in this pattern, not just the ones named train_*.
#
# 2026-08-20: and again, for src/ensemble.py. Naming the scripts one at a time
# has now failed three times in a row, each time the same way: something the
# queue itself launched was invisible to the check that decides whether a slot
# is free. Match on shape instead of on a list - the interpreter, then a script
# under src/ - so any future entry is covered without an edit. The optional -u
# rather than `.*` is deliberate: `.*` makes this pattern match a shell whose
# own command line merely quotes it (a diagnostic pgrep counts as a running
# job), which is the self-match that already cost 26 minutes once.
# 2026-08-25: the inverse of the three bugs above. Shape-matching any src/*.py
# also matches scripts the queue never launched and that never touch the GPU.
# src/ask.py sat waiting eight hours for a Telegram answer, the runner counted
# it as a running job, and the card stayed idle for 70 minutes with work queued.
# Enumerating GPU scripts failed three times, so keep matching on shape and
# subtract the known non-GPU helpers; anything new still defaults to blocking,
# which is the safe direction to be wrong in.
# 2026-09-19: and the self-match finally cost hours instead of minutes. A
# diagnostic wait loop - `until ! pgrep -f "\.venv/bin/python3 -u
# src/ensemble.py"` - was left running by a session that had moved on. Its own
# zsh command line contains the pattern, so it matched itself and never exited,
# and `pgrep -af` matched THAT shell here: this runner counted one job running
# from 13:46 to 17:10 with an idle card and three lines queued, and said nothing
# the whole time, because the busy branch is the silent one.
#
# Matching command lines cannot be made safe - anything that mentions the
# pattern is indistinguishable from the thing itself. Match the process NAME
# first (`pgrep -x`, which a shell called zsh or bash can never satisfy), then
# confirm through /proc/<pid>/exe that it really is one of our interpreters,
# and only then look at the arguments to decide whether it is a GPU job.
NOGPU='src/(ask|notify)\.py'
running() {
  local n=0 pid exe cmd
  # No /proc/<pid>/exe check: it is the resolved binary, not argv[0], so a run
  # launched as .venv/bin/python3 reports
  # ~/.local/share/uv/python/cpython-3.12.13-.../bin/python3.12 and no venv
  # pattern can match it. That version of this function counted 0 while
  # m91dinofrzs1 was training - which is the dangerous direction, since the
  # runner would have launched the next line straight onto a busy card.
  #
  # The name filter alone does the work that matters: `pgrep -x` matches the
  # process NAME, and a shell is called zsh or bash no matter what its command
  # line quotes. Everything reaching the loop body is already an interpreter.
  for pid in $(pgrep -u "$(id -u)" -x 'python[0-9.]*' 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$cmd" in *src/*.py*) ;; *) continue ;; esac
    echo "$cmd" | grep -qE "$NOGPU" && continue
    n=$((n + 1))
  done
  echo "$n"
}
free_mib() {
  local used total
  read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
                         --format=csv,noheader,nounits | head -1 | tr ',' ' ')
  echo $(( total - used ))
}


# Inputs a queue line reads that do not exist yet.
#
# This queue has no dependency tracking, and three times in a month a line that
# named an artifact an earlier line had not produced yet was launched anyway,
# died in under a second, and was popped off as though it had run: blend_targets
# and m29blend40 on 13 August, and three ensemble jobs in September. The failure
# is silent in the worst way - the exit-status check below does fire, but by then
# the line is gone from the queue and the next one is already on the card.
#
# Outputs are exempt by the flag in front of them, which is the only way to tell
# them apart from inputs: `--save-preds artifacts/x.npz` creates x.npz and
# `artifacts/x.npz` as a bare argument reads it. A flag not listed here makes its
# path look like an input, which defers a line that would have run - noisy, but
# the safe direction, unlike the reverse.
OUTFLAGS='--out|--output|--outdir|--out-dir|--save|--save-preds|--save-to|-o'
missing_inputs() {
  local prev="" tok
  for tok in $1; do
    if echo "$prev" | grep -qE "^($OUTFLAGS)$"; then prev="$tok"; continue; fi
    case "$tok" in
      artifacts/*|data/*) [ -e "$tok" ] || printf '%s ' "$tok" ;;
    esac
    prev="$tok"
  done
}

# The first queue line all of whose inputs exist. A line that is not ready is
# left in the queue rather than dropped: whatever produces its input may well be
# further down, and a run that is merely out of order should cost an ordering,
# not an experiment.
# Sets NEXT and DEFERRED rather than printing NEXT: `next=$(pick_next)` is a
# subshell, so DEFERRED set inside it would never reach the caller and every
# blocked line would be skipped in silence - the same silence this is here to
# remove. `while ... done < <(...)` keeps the loop in this shell, so the
# assignments survive.
pick_next() {
  local line miss
  DEFERRED=""; NEXT=""
  while IFS= read -r line; do
    miss=$(missing_inputs "$line")
    if [ -n "$miss" ]; then
      DEFERRED="$DEFERRED  ${line:0:70} needs: $miss
"
      continue
    fi
    NEXT="$line"
    return
  done < <(grep -vE '^\s*(#|$)' "$QUEUE")
}

echo "queue runner up $(date '+%F %T')  max_jobs=$MAX_JOBS free_min=${FREE_MIN}MiB"

# 2026-08-13: m32blend40new OOM'd against a process that no longer existed by
# the time anyone looked - the previous job (oof_predict) had genuinely
# exited, running() correctly saw 0, but the driver had not yet reclaimed its
# ~17GB of VRAM. free_mib() sampled during that reclaim window and read
# stale headroom. Track the running->idle transition and force one extra
# poll's wait before trusting free_mib() right after a job ends, rather than
# after every poll (which would just slow the steady-state loop for nothing).
was_running=0
waited=0
skipnote=0

while true; do
  pick_next; next="$NEXT"
  now_running=$(running)
  if [ "$was_running" -gt 0 ] && [ "$now_running" -eq 0 ]; then
    echo "$(date '+%T') last job exited - one extra poll before trusting free VRAM"
    was_running=$now_running
    sleep "$POLL"; continue
  fi
  was_running=$now_running

  if [ -z "$next" ]; then
    if [ "$(running)" -eq 0 ]; then
      if [ -n "$DEFERRED" ]; then
        # Not idle-and-done. Every queued line is waiting on a file nothing is
        # going to write, so say which - this state used to look identical to an
        # empty queue and would sit here until someone read the file by hand.
        echo "$(date '+%T') GPU idle and every queued line is BLOCKED on a missing input:"
        printf '%s' "$DEFERRED"
      else
        echo "$(date '+%T') queue empty and GPU idle - nothing left to run"
      fi
    fi
    sleep "$POLL"; continue
  fi
  # Its own counter, not `waited`: waited is zero whenever the queue is moving,
  # so keying off it would reprint this every single poll.
  if [ -n "$DEFERRED" ]; then
    if [ $((skipnote % 30)) -eq 0 ]; then
      echo "$(date '+%T') skipping $(printf '%s' "$DEFERRED" | grep -c .) line(s) waiting on inputs:"
      printf '%s' "$DEFERRED"
    fi
    skipnote=$((skipnote + 1))
  else
    skipnote=0
  fi

  # Say something while waiting. This branch was silent, and silence here is
  # indistinguishable from a healthy long run: the 3h22m stall on 19 Sep looked
  # exactly like a training job in progress from the log alone. Once every ten
  # polls, name what is holding the slot, so a wrong answer is visible without
  # anyone running ps.
  if [ "$(running)" -ge "$MAX_JOBS" ] || [ "$(free_mib)" -lt "$FREE_MIN" ]; then
    waited=$((waited + 1))
    if [ $((waited % 10)) -eq 1 ]; then
      echo "$(date '+%T') waiting: $(running) job(s), $(free_mib)MiB free, "\
           "$(grep -cvE '^\s*(#|$)' "$QUEUE") queued - holder: "\
           "$(pgrep -u "$(id -u)" -x 'python3?' -a 2>/dev/null | grep -m1 'src/' | cut -c1-90)"
    fi
    sleep "$POLL"; continue
  fi
  waited=0

  # Pop before launching. If this script dies between the two, the experiment is
  # lost - but a duplicate launch would corrupt a checkpoint directory that a
  # live run is writing to, which is worse and harder to notice.
  tag=$(echo "$next" | grep -oE '\-\-tag [A-Za-z0-9_]+' | awk '{print $2}')
  tag=${tag:-run$(date +%s)}
  grep -vxF "$next" "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  echo "$(date '+%F %T')  $next" >> "$STARTED"
  echo "$(date '+%T') launching $tag  (free $(free_mib)MiB, running $(running))"
  # Run through a wrapper that records the exit status in the log. Guessing
  # liveness from log *contents* only ever worked for trainers: see the health
  # check below.
  # ( $next ) in a subshell: a queue line that ends in `exit` would otherwise
  # take the wrapper down with it and the status line would never be written.
  nohup bash -c "( $next ); echo \"[queue] exit \$? after \$SECONDS s\"" \
        > "$LOGDIR/train_$tag.log" 2>&1 &
  jobpid=$!
  launched=$(date +%s)
  # Wait for the interpreter to actually appear, not a fixed guess. `uv run`
  # resolves the environment before exec'ing python, so a flat sleep can expire
  # while running() still reads 0 - which launches the next job on top of this
  # one. Bounded so a job that dies at startup cannot wedge the queue.
  # 2026-09-06: break on the job EXITING too, not only on it appearing. The
  # wait was written for trainers, which take minutes to reach the interpreter
  # and hours to finish. A measurement job that finishes faster than the
  # detection window never makes running() non-zero, so the loop spun the full
  # 60 x 2s and the runner sat for 140 seconds after a job that took 1 - which
  # is most of what src/merge_preds.py, src/ensemble_search.py and the probes
  # cost. Four short jobs in a row is a quarter-hour of the card doing nothing.
  for _ in $(seq 60); do
    [ "$(running)" -gt 0 ] && break
    kill -0 "$jobpid" 2>/dev/null || break
    sleep 2
  done
  sleep 20          # let it claim its memory before the next free_mib check

  # Report on how a job actually ended, not on whether its log looks like a
  # trainer's. The OOM killer sends SIGKILL, so a run that asks for too much RAM
  # leaves no traceback at all - it just stops mid-log, and the queue cheerfully
  # starts the next one on top of the same fault. Three experiments were consumed
  # that way on 2026-08-09 before anyone noticed the logs ended after "cosine
  # schedule:".
  #
  # 2026-08-21: the previous version tested `grep "^ep "` plus a pgrep on
  # `--tag $tag`, and both halves were wrong for anything that is not a training
  # run. ensemble.py prints no epoch lines and carries no --tag, so every one of
  # the six ensemble measurements on 20-21 Aug was flagged "suspect OOM" after
  # completing perfectly. A warning that fires on healthy jobs is worse than no
  # warning: it is how a real OOM gets read as more of the same noise.
  #
  # The wrapper above appends "[queue] exit N" to the log, so ask the exit status
  # instead of the log's shape - and wait for the actual exit rather than
  # sampling at 300s, which never saw a job that died in hour three of five.
  ( while kill -0 "$jobpid" 2>/dev/null; do sleep 10; done
    elapsed=$(( $(date +%s) - launched ))
    status=$(grep -o '^\[queue\] exit [0-9]*' "$LOGDIR/train_$tag.log" 2>/dev/null \
             | tail -1 | awk '{print $3}')
    if [ -z "$status" ]; then
      # The wrapper itself never got to print - it was killed too.
      echo "$(date '+%T') WARNING $tag vanished after ${elapsed}s with no exit status"
      echo "  last lines: $(tail -2 "$LOGDIR/train_$tag.log" 2>/dev/null | tr '\n' ' ')"
      echo "  suspect OOM - SIGKILL leaves no traceback. check dmesg -T | grep -i oom"
    elif [ "$status" != "0" ]; then
      echo "$(date '+%T') WARNING $tag exited $status after ${elapsed}s"
      echo "  last lines: $(tail -2 "$LOGDIR/train_$tag.log" 2>/dev/null | tr '\n' ' ')"
      [ "$status" = "137" ] && echo "  exit 137 is SIGKILL: check dmesg -T | grep -i oom"
    fi ) &
done
