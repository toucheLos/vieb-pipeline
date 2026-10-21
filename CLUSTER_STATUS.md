# CLUSTER_STATUS: VIEB mouse, last two weeks of cluster access

Day 0 is **2026-10-21**. **Deadline: not given**, so the prompt's default is
assumed: **14 days, 2026-11-04** (owner, 2026-10-21). **No new jobs from
2026-11-02.** The owner is mostly unreachable from **2026-10-30**. After that,
an open decision takes its default below, or that branch is parked here and the
rest continues.

The governing prompts `MOUSE_PROGRAM_PROMPT.md`, `MOUSE_AGENT_PROMPT_2.md` and
`PAPER1_PROMPT.md` are **not on disk**. Their rules are carried by the
registrations, `results/DEVIATIONS.md`, `results/STATUS.md` and this file.

## Decisions (§7)

| Decision | Value | Source |
|---|---|---|
| Deadline | 2026-11-04 (assumed: 14 days) | owner, 2026-10-21 |
| Destination, code | new **private GitHub repos** under `toucheLos` for `recur`, `shapeflow`, `vieb-atlas` and `vieb2`. `vieb-pipeline` already pushes to `toucheLos/vieb-pipeline` | owner, 2026-10-21 |
| Destination, bulk results and lab data | the owner's own machine. **The owner pulls** with rsync over SSH through the login node, since there is no data-transfer node or Globus. Lab data goes to a separate restricted directory on that host | owner, 2026-10-21 |
| Lab approval to copy lab data | **stated by the owner as given (2026-10-21)**. The approver's name and the approval document: **TO BE ENTERED BY THE OWNER** | owner |
| Max concurrent mouse GPU jobs | **9** (half of 18). **3 during the BJJ re-score window, 2026-10-29 to 10-31** (`~/bjj/CLUSTER_STATUS.md`) | default |
| Mouse GPU job priority | `--nice=1000` on every mouse job | prompt |
| V-JEPA model and tune grid | `facebook/vjepa2-vitg-fpc64-256`, **full token grid on tune** | owner, 2026-10-21 |
| Density re-run | only if GPUs are idle in the last week | default |
| V-JEPA beyond CFD days | yes, if the budget allows | default |

**Owner actions before 2026-10-30:**
1. Create four empty private GitHub repos: `recur`, `shapeflow`, `vieb-atlas` and `vieb2`.
2. Enter the lab-approval record above.
3. Pull and `verify` batches b1 and b2.
4. Run `repro_check.sh` once on the exported copy.

## Benchmarks and budget

(Filled in from the Day 0–1 benchmark job; see below.)

## Copy-out ledger (§0)

A batch is out only when the owner's `verify` run reports PASS.

| Batch | Content | Files | Size | Prepared | Verified |
|---|---|---|---|---|---|

## Jobs

| Job | What | State |
|---|---|---|

## Parked branches

(none yet)
