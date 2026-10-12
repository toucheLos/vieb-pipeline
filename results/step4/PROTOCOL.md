# STEP 4 §4.0 — protocol facts: **STOP**, the required facts are not recorded

The prompt requires four things before Step 4 may be registered: each
session's day and context, when shocks occur, and which sessions are
conditioning, test or extinction. Each must have a source in `labels.py` or
the metadata. **Protocol is not to be inferred from behaviour.** Of these, only
day and context are recorded.

## What is recorded

`recur/recur/labels.py` parses every field from the recording ID, and cites no
protocol document:

```
20241016_Box_1_CFC_Day_0_(Context_A)_308
^date    ^box  ^proto ^day ^context   ^animal
```

| session | n recordings | recorded |
|---|---|---|
| CFC day 0, Context A | 298 | day, context, protocol string `CFC` |
| CFC day 1, `Context_A,_No_Shock` | 298 | the **only** explicit shock tag in the corpus |
| CFC day 2, Context C | 298 | day, context |
| CFD days 3–6, Context A and Context B | 298 each | day, context, protocol string `CFD` |
| CFD day 7, A and B | 284 each | day, context |

## What is missing

| fact Step 4 needs | status | evidence |
|---|---|---|
| **When shocks occur** (time in session, count, intensity) | **not recorded anywhere** | `recur/labels.py` docstring: "Which of the remaining sessions delivered shock is **not recoverable from the id**", and a `shocked` flag is refused. `vieb/vieb_v2/scripts/moseq_control.py:30`: "the shock timestamps are not in this data" |
| **Which sessions delivered shock** (day 0? CFD-day A sessions?) | **not recorded** | only `No_Shock` is tagged. `vieb/vieb_v2/transfer/stage8_moseq_control.py:128` defaults `shock_day=0, context="A"`, an undocumented default and not a source |
| **Session type** (conditioning / test / extinction) | **not recorded** | only `CFC`/`CFD` and the day number. Reading CFC as conditioning and CFD as discrimination would be an inference from the acronyms |
| **Which context is paired with shock on CFD days 3–7** | **not recorded** | — |
| A fear or freezing score | the column exists and is empty | `vieb/metadata.csv` `fear` |

## What would unblock Step 4

**Ask Luna lab for the protocol sheet:**
- shock times per session;
- which sessions are conditioning, test and extinction;
- which of Context A and Context B is the shock-paired context on CFD days.

Once that sheet is committed with its source (in `labels.py`, or a metadata
file it reads), Step 4 can be registered.

The same sheet answers the prompt's §7 question, "whether the
context-to-shock mapping may be named". Until it exists, the contexts stay
"Context A" and "Context B".
