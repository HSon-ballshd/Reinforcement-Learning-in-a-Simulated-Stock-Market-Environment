# Collaboration Hub — AIE301M Project

This folder is the shared workspace for three Opus agents collaborating on the AIE301M
project defined in `../Aie Project Overview.md`.

If you are an agent: **read PROTOCOL.md first, then SPEC.md, then CONTRACT.md.**
Before starting any work, also read STATUS.md so you don't duplicate effort.

## Files

| File | What it is | When to read |
|---|---|---|
| [PROTOCOL.md](PROTOCOL.md) | Rules of engagement. Roles, ownership, handoffs, conflict resolution. | **Always, first.** |
| [SPEC.md](SPEC.md) | Restated, agent-facing version of the project spec (objectives, hypotheses, evaluation parameters, scope). | After PROTOCOL.md. |
| [CONTRACT.md](CONTRACT.md) | Interface contract. Exact file paths, function signatures, and data formats each module must expose. | Before writing or modifying any code. |
| [STATUS.md](STATUS.md) | Live status board. Who's working on what, in-progress, blocked, decisions log, results summary. | Before starting work; update when you finish or block. |

## Folder ownership (exclusive)

Each agent only edits files inside their own folder. Cross-folder changes go through
the interface defined in CONTRACT.md and a handoff entry in STATUS.md.

| Agent | Folder | Responsibility |
|---|---|---|
| **Sim** | `sim/` | Market simulator (Cookie Clicker dynamics port). |
| **Eval** | `eval/` | Regime classifier + DQN trader + baselines + evaluation harness. |
| **Lead** | _(no code folder)_ | Reads all, writes only `collaboration/`. Sets next-step instructions, arbitrates conflicts, integrates. |

`sim/` and `eval/` both **read from** `collaboration/CONTRACT.md` and **read from**
the other agent's published artifacts (paths in CONTRACT.md). They do not write to
each other's folders.
