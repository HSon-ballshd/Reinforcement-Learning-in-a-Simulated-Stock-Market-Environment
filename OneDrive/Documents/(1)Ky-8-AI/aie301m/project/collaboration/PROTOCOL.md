# Protocol — 3-Agent Collaboration Rules

This document is the **rulebook** for three Opus agents collaborating on the AIE301M
project. Read it before doing anything else.

## 1. Roles

There are three roles, each staffed by one Opus agent:

- **Sim** — owns the market simulator. Source of truth for *what the market does*.
- **Eval** — owns the regime classifier, the DQN trader, the heuristic baselines, and the evaluation harness. Source of truth for *how the agents are measured*.
- **Lead** — owns nothing in `sim/` or `eval/`. Owns only this folder. Reads both, sets priorities, writes next-step instructions, arbitrates conflicts, integrates results.

Why these three and not, say, three implementers? Because splitting "build the world"
from "play in the world" from "decide what to do next" is the cleanest split for this
project. Eval cannot grade what Sim has not built. Lead cannot direct what neither has
shipped. Each role has a single, non-overlapping mandate.

## 2. Folder ownership (exclusive write access)

Each role only writes to its assigned folders. This is non-negotiable.

| Role | May write to | May read from |
|---|---|---|
| Sim | `sim/**`, `tests/test_sim*` | everything in the project |
| Eval | `eval/**`, `tests/test_eval*`, `tests/test_integration*` | everything in the project |
| Lead | `collaboration/**` only | everything in the project |

If a role needs a change in another role's folder, it does **not** make the edit
itself. It files a handoff in STATUS.md (see §6) and waits.

## 3. Source of truth

There are four shared artifacts. In order of authority:

1. `../Aie Project Overview.md` — human-authored project spec. Top of the chain.
2. `SPEC.md` — agent-facing restatement. If it ever disagrees with (1), the human
   decides; until then, (1) wins.
3. `CONTRACT.md` — interface contract. Defines exactly how `sim/` exposes the market
   to `eval/`. Both Sim and Eval must conform.
4. `STATUS.md` — live state. Who is doing what, what is blocked, what was decided.

Lead owns (2), (3), (4). Sim and Eval may **propose** edits to (3) via STATUS.md, but
only Lead merges them.

## 4. Boot sequence (every agent, every session)

Before you touch any code, in this order:

1. Read this PROTOCOL.md.
2. Read SPEC.md.
3. Read CONTRACT.md.
4. Read STATUS.md — especially the "In Progress" and "Blocked" sections.
5. Confirm your role's folder is the one you expect to edit.
6. State your one-sentence plan in chat before writing code. ("I will add
   `CookieClickerMarket.tick()` to `sim/market_sim/simulator.py` to match the JS lines
   815–820.")

If you skip this, you will step on the other agents' work. Don't skip it.

## 5. Task lifecycle

Every task moves through five states. Update STATUS.md at each transition.

```
proposed  →  claimed  →  in_progress  →  review  →  done
                                    ↘  blocked
```

- **proposed** — anyone (incl. Lead) puts it in STATUS.md under "Proposed."
- **claimed** — the role that will do it adds their name and moves it to "Claimed."
- **in_progress** — moved to "In Progress" when actual editing starts.
- **review** — done from the author's side; Lead checks it conforms to CONTRACT.md
  and integrates it. Other roles may read but not edit.
- **done** — Lead merges. Task moves to "Done (this session)" with the commit hash.
- **blocked** — author is stuck. Lead or another role must unblock within one session
  turn, or the human is asked.

A role may only have **at most 2 tasks in `in_progress` simultaneously**. If you want a
third, finish or block one first. This prevents context drift.

## 6. Handoffs

A handoff is when one role needs something from another. Format in STATUS.md:

```
HANDOFF <id>
FROM:    <role>
TO:      <role>
NEEDS:   <one sentence>
WHY:     <one sentence>
ACCEPT:  <test or interface point that proves it works>
```

The receiving role claims the handoff within one turn, or escalates to Lead. Never
silently ignore a handoff.

## 7. Conflict resolution

Two agents disagreeing about a design decision (e.g. feature set, evaluation metric,
reward shaping).

1. Each side writes a 3–5 line position in STATUS.md under "Disputes."
2. The third agent (the one not involved) reads both and picks one, with a one-line
   reason, also in STATUS.md.
3. That's the decision. The losing side does not relitigate in the same session.
4. If all three are involved, or the third abstains, **defer to a human**. Lead posts
   the question to the human and pauses the disputed work until answered.

Rationale: two-of-three majority is the fastest path that still preserves a check. The
human escape hatch exists for cases where the disagreement is about project goals,
not implementation details.

## 8. Version control & commits

- One commit per task reaching `done`. Commit message format:
  `<role>: <verb> <thing> (<task-id>)`. Example: `sim: add tick() to CookieClickerMarket (T-003)`.
- Never force-push. Never amend a commit the other agents have already pulled.
- Lead squashes when integrating across roles, in a separate commit:
  `lead: integrate <feature> (<task-id>)`.
- Branches: stay on `main` for now. If a session needs isolation, use a feature
  branch named `<role>/<short-desc>` and merge via PR. Lead approves the PR.

## 9. Tests

- Every PR includes tests for the new code. No exceptions.
- Sim tests live in `tests/test_sim*.py`. Eval tests in `tests/test_eval*.py`.
  Integration tests (sim ↔ eval) in `tests/test_integration*.py`.
- `pytest` from the project root must pass before a task moves to `done`.
- If a test fails and the cause is in another role's folder, file a handoff. Do not
  patch around it by editing the other role's folder.

## 10. Communication style

- Be terse. Each STATUS.md entry fits on a few lines.
- No "I think maybe we could possibly consider..." — say the decision or say you don't know.
- Numbers, not adjectives. "Mean regime-3 price after 1000 ticks is 47.3" not
  "regime 3 looks pretty bullish."
- Cite the file:line you're changing. "Editing `sim/market_sim/simulator.py:42` to
  match `minigameMarket.js:815`."
- If you read something in another agent's folder that looks wrong, **say so in
  STATUS.md** under "Concerns," don't fix it unilaterally.

## 11. When you are stuck

1. Re-read PROTOCOL.md. (Yes, really.)
2. Re-read CONTRACT.md. Your contract assumption may be wrong.
3. Check STATUS.md for a relevant handoff or decision you missed.
4. If still stuck after 2 turns of real work, mark the task `blocked`, write what you
   tried and what you need, and stop. Lead or human will unblock you.

## 12. What Lead does NOT do

- Lead does not write simulation code.
- Lead does not write training code.
- Lead does not write tests.
- Lead writes the plan, the contract, the spec, the status board, the decision log,
  and the human-facing summary.

If Lead catches itself editing code, that's a smell. Step back and write a task for
Sim or Eval instead.
