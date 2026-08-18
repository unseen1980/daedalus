# Boot resume, proven through the installed service

2026-08-11 05:38–05:42 UTC. Cost: nothing — no GPU, arm 2 training throughout.

## What was missing

`hero` is 133 h on a rented box. `supervise.run_with_resume` restarts a crashed
`train.py`, but it is a loop *inside* the launcher process, so a stop/start or
host reboot takes the supervisor down together with the run. Nothing on this
instance started either again — checked, not assumed:

    /root/onstart.sh                        one line, `entrypoint.sh`
    crontab -l                              "no crontab for root"
    /etc/supervisor/conf.d/                 8 stock services, none ours
    vast-capabilities .workspace_is_volume  False

The box would come back, rent at $0.449/h and run nothing until a human looked.
The 2026-08-08 wedge ended in exactly that state, ~$0.85 and two hours. With
~$7 of buffer, one unnoticed 15.6 h restart is all of it.

## What the live check exercised

The unit and end-to-end tests cover `boot_resume.py`. This covers the one
composition they cannot: **supervisord → wrapper → boot_resume → run_with_resume
→ a child process**, on the real box, with a real marker, through the service as
installed.

Arranged the state a reboot leaves — an open `inflight.json`, a checkpoint
recording step 91,337, and a `train.pid` holding a stale pid — then
`supervisorctl start daedalus_resume`. From `/var/log/portal/daedalus_resume.log`:

    [boot-resume] settling for 60s
    [boot-resume] rearm heartbeat: already running
    [boot-resume] rearm credit_watch: already running
    [boot-resume] run       : boot-resume-live-check
    [boot-resume] checkpoint: runs/.../checkpoint.pt (present)
    [boot-resume] boot resumes so far: 0/5
    [boot-resume] boot resume #1
    [supervise] attempt 1/10 (resuming): ... --resume runs/.../checkpoint.pt
    [boot-resume] finished: {"attempts": 1, "resumed": true, "returncodes": [0]}

What the child actually received, read back from its own process rather than
inferred:

```json
{"argv": ["--run-name", "boot-resume-live-check", "--resume",
          "runs/boot-resume-live-check/checkpoint.pt"],
 "resumed": true, "step_at_start": 91337}
```

Marker afterwards: `completed: true`, `outcome: "completed"`, `boot_resumes: 1`.
Cleaned up; `--dry-run` then reported nothing in flight. Arm 2 unaffected
throughout.

## The line it exists to prove

`attempt 1` carried `--resume`. `run_with_resume` normally adds it only from
attempt 2, which is right for a fresh launch and wrong for a restart: attempt 1
of a *new* supervisor over a 90-hour-old run would relaunch clean, start at step
0, and overwrite the rolling checkpoint on its first save — destroying the run
while looking healthy from outside. `force_resume=True` is what makes the line
above read `(resuming)`, and reverting it fails
`test_end_to_end_a_simulated_reboot_continues_from_the_checkpoint` with "child
got no --resume".

## What this does not cover

- An actual reboot. The instance must never be rebooted deliberately
  (AGENT.md §0.1), so the boot linkage is established by inspection:
  `/etc/supervisor/supervisord.conf` has `files = /etc/supervisor/conf.d/*.conf`
  and the conf sets `autostart=true` — the same mechanism that brings back the
  8 stock services, all showing 2-day uptimes from container start.
- A **recycle or destroy**. `/etc/supervisor/conf.d` is container filesystem and
  `workspace_is_volume` is False, so both are wiped. Recovery there is the Hub
  checkpoints, and the service must be reinstalled with
  `bash scripts/install_resume_service.sh`.
- `abl_arch.py`, which carries its own copy of the retry loop and writes no
  marker. `hero.py` goes through `run_with_resume` and does.
