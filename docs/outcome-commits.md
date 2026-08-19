# Outcome commits: git-style compaction of a conversation

Status: DESIGN, not implemented. Companion to `.tasks/kv-context-budget-with-reasoning.md`
and engram id:516/id:517.

## The problem this solves

A conversation's KV grows by roughly the length of every answer. Measured on DSv4-Flash
(2026-08-19): canonical grows about 120 net tokens per turn on substantive Q&A, and `n_gen` runs
159-2048. The window is enforced by an admission check — `prompt + n_predict > n_ctx` is refused
outright — so a long conversation does not degrade, it stops being accepted. Before that, the
turns come back EMPTY: on a thinking model the token cap gets spent entirely on reasoning and no
content is emitted at all (2 of 7 substantive turns, at a 2048-token cap).

Both failures are quiet. The fix is to stop carrying the full text of settled work.

## The idea

Separate WORK IN PROGRESS from OUTCOMES, the way a repository separates a working tree from its
commit history.

* The conversation since the last commit is the working tree: full answers, exploration, dead ends.
* A commit collapses that span into ONE outcome message, chosen by the human.
* What the model sees is the commit log plus the uncommitted tail.

The reasoning span is the degenerate case already shipped: reasoning is work in progress, the
answer is the outcome, and the answer is what reaches the canonical sequence
(`kv-canonical-scratch-sequences.md`). This design lets the outcome be SMALLER than the answer.

## What it costs in the KV, and why this shape and not another

A commit rewrites the TAIL of the conversation. That matters because of what this engine's memory
can and cannot do:

* An INTERIOR edit — removing a settled span from the middle while keeping what follows — is a
  positional cut. A recurrent memory (Gated Delta Net, DSv4-Flash) refuses it, and the surviving
  block would in any case have attended to text that is no longer there (`kv-prefix-reuse-
  recurrent-memory.md`, engram id:498).
* A TAIL rewrite needs no cut at all. Everything up to the last commit is untouched; the compacted
  history is laid down again from there.

So "commit at the tip, never rewrite history" is not a borrowed git metaphor — it is the only
shape this memory supports. Git's constraint and the engine's constraint happen to coincide.

## The engine change required: NONE, for the first version

This is the part worth stating plainly, because it was not obvious.

`session.cpp` already replaces `chat_history` with the caller's messages and, when that history is
not an extension of the canonical sequence, drops canonical and lays it down again from zero. A
client that replaces N messages with one outcome message is exactly that case. The engine needs no
new field, no new sequence, and no knowledge that a commit happened.

The cost is ONE prefill of the compacted history per commit — and the compacted history is short
BY CONSTRUCTION, which is the whole point of committing. Commits are human-triggered and rare.

A `canonical_override` on the request plus the committed form on the response would remove even
that prefill, and a fourth sequence holding the commit point would remove it again. Both are
OPTIMISATIONS to be justified by measurement, not prerequisites. Do not build them first.

## The UX

Drafting is assisted; committing is not. The model proposes, the human disposes.

* `/commit` — the model drafts the outcome, the human edits and accepts. Nothing auto-commits.
* The preview is a diff of COST, not of text:
  `7 turns, 3,400 tokens -> outcome, 180 tokens. Window 71% -> 12%.`
  The trade is visible before it is made.
* `/amend` — re-commit replaces the last outcome. Free: another tail rewrite.
* Status: `12 turns, 4.2k uncommitted, 68% of window`. This number is invisible today, which is
  why the wall arrives as a surprise.
* `/log` — the outcomes in order. That IS the conversation the model sees.

At ~70% of the window, OFFER a commit. Never fire one. The human decides when a span is settled;
that judgement is the feature.

## Safety: nothing is lost, only moved

The client keeps the FULL RAW TRANSCRIPT on disk. Only the KV holds outcomes. Recovering a
compacted span means re-injecting it and paying one prefill.

That converts "did the summary drop something I needed?" from data loss into a cost question, and
it is what makes unsupervised drafting acceptable to review rather than frightening.

## The failure mode to design against

A tidy summary that silently drops a NEGATIVE result. "We tried X, it does not work" is
first-class knowledge (engram id:315); drop it and the model cheerfully proposes X again — a worse
and quieter failure than running out of context, because it costs a whole turn and looks like
progress.

Therefore:

* the drafting prompt states explicitly that tried-and-refuted findings, dead ends and
  "does not exist" facts must survive the summary;
* the preview surfaces what was DROPPED, not only what was kept.

## Reaching any agent (a later step, not this one)

An arbitrary OpenAI-compatible agent sends its full transcript every turn, so an engine-side
compaction it does not know about would diverge from what it sends and re-prefill everything. Two
ways out:

* **A — cooperating client.** It stores the committed form and sends that back. This design.
* **B — translating engine.** The engine hashes each message and keeps a map from the raw turns to
  the outcome that replaced them, substituting on the way in. Any agent then works unmodified and
  the engine quietly serves a compacted KV.

B is the more powerful product and it is additive — but it removes the human from the loop, since
for an arbitrary agent the engine must decide what and when to compact. Build A first: the
drafting discipline and the negative-results rule get proven with somebody checking them before
they are ever allowed to run unattended.

## What is open

1. **Committing mid-thought.** Committing a PREFIX of the uncommitted tail and keeping the rest is
   an interior edit again. The honest answer is commit-all-or-nothing at the tip; if a boundary is
   wanted, commit before starting the next thing. Confirm this is acceptable in use before
   designing anything cleverer.
2. **Who drafts.** The same model that is answering (cheap, in-session, but it is the model whose
   context we are trying to shrink) or a smaller one. Unmeasured.
3. **Quality effect.** The model stops seeing its own full answers. This is a BEHAVIOURAL change,
   not a KV one, and it is the only risk here that a trace cannot settle — it needs an A/B.

## Verification

* A conversation driven to the admission-check refusal, then committed, accepts turns again.
* Prompt tokens after a commit reflect the compacted history, and the turn AFTER the commit reuses
  its prefix (`tail=0`, no refusal warning).
* A commit whose span contains a refuted approach produces an outcome that still names it; the
  model does not re-propose it afterwards.
* The raw transcript on disk still contains every turn the outcome replaced.
