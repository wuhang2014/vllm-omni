---
name: subagent-delegation
description: Use this skill whenever a task is large, multi-step, or risks exceeding the context window — such as refactoring multiple files, analyzing a large codebase, running a long series of edits, generating extensive documentation, or any task that could produce or consume a lot of tokens. Automatically decomposes the work into independent subtasks and delegates each to a separate subagent so that no single agent accumulates too much context. Prefer more subagents over fewer. Trigger on: "refactor", "analyze the whole", "across all files", "generate docs for", "migrate", "audit", or any request touching more than 2-3 files at once.
---

# Subagent Delegation Skill

## Purpose

This skill prevents context window exhaustion by decomposing large tasks into small, scoped subtasks and delegating each to its own subagent. Each subagent works in isolation with a minimal context slice, then reports results back. This keeps every agent well within its token budget.

**Golden rule: more subagents is always better than fewer. When in doubt, split further.**

---

## When to Apply This Skill

Apply this skill proactively whenever:
- The task touches more than 2–3 files
- The task involves reading, editing, or generating large amounts of text or code
- The task is sequential but each step is independent (e.g. lint → test → document)
- A previous attempt hit or approached a context limit
- The user uses words like: refactor, audit, migrate, analyze, document, scan, generate, across all, entire codebase

---

## Decomposition Strategy

Before delegating, break the task into the **smallest independently executable units** possible.

### Step 1 — Scope Inventory
List every file, module, or concern the task touches. Do this before writing any code or output.

```
Files/modules in scope:
- src/auth/login.ts
- src/auth/session.ts
- src/api/users.ts
- ...
```

### Step 2 — Identify Natural Split Points
Split by:
- **File or module** — one subagent per file is ideal for edits
- **Concern** — e.g. types, logic, tests, docs as separate subagents
- **Pipeline stage** — e.g. lint, refactor, test, document as separate passes
- **Domain** — e.g. auth, payments, UI as separate subagents

### Step 3 — Write a Delegation Plan
Before spawning any subagent, output the full plan so the user can see the breakdown:

```
Delegation Plan
───────────────
Subagent 1: Refactor src/auth/login.ts — extract token logic into helpers
Subagent 2: Refactor src/auth/session.ts — align session shape with new token helpers
Subagent 3: Update src/api/users.ts — consume updated auth API
Subagent 4: Regenerate docs/auth.md — document the new auth flow
Subagent 5: Run and fix failing tests in tests/auth/
```

### Step 4 — Spawn Subagents Sequentially or in Parallel

**Sequential** (when subagent N depends on subagent N-1's output):
- Pass only the minimal output of the previous subagent as input to the next
- Never carry the full conversation history forward

**Parallel** (when subtasks are fully independent):
- Spawn all subagents simultaneously
- Merge results at the end

---

## Subagent Instructions Template

Each subagent must be given a tightly scoped prompt. Use this template:

```
You are a focused subagent. Your only job is the task below.
Do NOT attempt anything outside this scope.

## Your Task
<single, specific task — e.g. "Refactor src/auth/login.ts to extract token validation into a helper function called `validateToken`">

## Input
<only the minimal context this subagent needs — the relevant file(s) or excerpt>

## Output
<exactly what to return — e.g. "Return only the updated file content, nothing else">

## Constraints
- Do not read or modify files outside your scope
- Do not summarize what you did — just return the output
- If you cannot complete the task without more context, say so and stop
```

---

## Context Hygiene Rules

1. **Never pass full conversation history to a subagent.** Strip it to the minimum needed.
2. **Never accumulate subagent outputs in a single growing context.** Store intermediate results as files or structured notes, not inline text.
3. **Prefer file I/O over in-context passing.** Write subagent output to a temp file; the next subagent reads only that file.
4. **One concern per subagent.** If a subagent needs to do two things, split it into two subagents.
5. **Summarize, don't carry.** When a subagent finishes, extract only the key result (e.g. a diff or a filename) — never the full output — before starting the next.

---

## Aggregation

After all subagents complete:
- Collect only the final artifacts (files, diffs, summaries)
- Produce a brief consolidated report:

```
Summary
───────
✓ Subagent 1: login.ts refactored — validateToken extracted
✓ Subagent 2: session.ts updated — shape aligned
✓ Subagent 3: users.ts updated — consuming new auth API
✓ Subagent 4: docs/auth.md regenerated
✓ Subagent 5: 3 failing tests fixed

Next steps: review diffs in /tmp/subagent-outputs/ before committing
```

---

## Anti-Patterns to Avoid

| Anti-pattern | Why it's bad | Fix |
|---|---|---|
| One agent does everything | Blows context | Split into ≥1 subagent per file |
| Passing full chat history to subagents | Wastes tokens | Pass only the scoped input |
| Accumulating all outputs in one place | Grows context fast | Write outputs to files |
| Vague subagent tasks | Agent over-reaches | One sentence, one file, one job |
| Waiting to split until context is full | Too late | Split proactively at plan time |

---

## Example: Codebase Refactor

**User request:** "Refactor all files in `src/` to use the new logger API"

**Wrong approach:** Read all files, edit them all in one pass → context explosion.

**Right approach:**
1. List all files in `src/` → 11 files found
2. Spawn 11 subagents, one per file, each with only that file's content
3. Each subagent returns only the updated file
4. Aggregate: write all 11 outputs to disk, report done

---

## Example: Documentation Generation

**User request:** "Generate API docs for the entire `src/api/` directory"

**Right approach:**
1. List files → 8 route files
2. Spawn 8 subagents, one per route file
3. Each returns a markdown doc section
4. A final aggregator subagent merges the 8 sections into `docs/api.md`

---

Always prefer more subagents. A task that could be done in 3 subagents should use 6 if there's any risk of context pressure. Splitting is cheap; hitting the context limit is not.
