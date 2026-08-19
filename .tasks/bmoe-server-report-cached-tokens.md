# bmoe-server reports no cached-token count, so reuse is invisible to a client

column: Todo
created: 2026-08-19

## Why it matters now

Prefix reuse on a thinking model was just fixed (`kv-canonical-scratch-sequences.md`), and the only
way anyone saw it work was `BMOE_KV_TRACE=1` on the server's stderr. A client sees nothing:
bmoe-server's `usage` has no `prompt_tokens_details.cached_tokens`, so bmoe-chat displays `kv 0
reuse` on a conversation that is reusing its prefix perfectly.

That is worse than cosmetic. engram id:489 records a session where reuse was ASSUMED and was not
happening, and the thing that caught it was reading `cached_tokens` per turn. The practical rule
from that measurement is "never assume reuse, read the meter" — and on this server there is no
meter to read.

The Python bridge already emits this. `~/.local/bin/bmoe-bridge.py` was patched on 2026-08-17 to
report llama-server-shaped usage: `prompt_tokens_details.cached_tokens` plus a `bmoe` block
carrying `n_prompt`, `n_past`, `prefill_s` and `tok_s`, on the streaming path too, before `[DONE]`.
Retiring the bridge in favour of the native server currently means losing that.

## What to do

* Report `usage.prompt_tokens_details.cached_tokens` on `/v1/chat/completions`, non-streaming and
  streaming alike. The number is the reused prefix — `n_common` in `session.cpp`, which the engine
  already computes and which `RunSummary` can carry out.
* Match the bridge's shape rather than inventing one, so bmoe-chat needs no client change and the
  two serving paths stay comparable while both exist.
* Consider carrying the `bmoe` block too (`n_prompt`, `n_past`, `prefill_s`, `tok_s`) for the same
  reason. Check what bmoe-chat actually reads before deciding how much of it is load-bearing.

## Acceptance

1. A 3-turn append-only conversation against bmoe-server shows a rising `cached_tokens` per turn,
   and the numbers agree with what `BMOE_KV_TRACE=1` prints for the same run.
2. bmoe-chat's reuse display is correct against bmoe-server with no client-side change.
3. `cached_tokens` is 0 on turn 1 and on a request whose history is not an extension — a wrong
   meter here is worse than none, since the whole point is catching reuse that is not happening.
4. The streaming path reports it too, before `[DONE]`.
5. ctest green on evo.

## Notes

* Verify against qwen3moe first (8s load, ~1s turns) and confirm on DSv4 once; the numbers to
  expect for both are in `kv-canonical-scratch-sequences.md`.
* Commits: Conventional Commits, author Helldez only, no AI co-author or session trailers.

## Related

* `.tasks/bmoe-server-drops-conversation-history.md` — where this was first noted as a leftover
* `.tasks/kv-canonical-scratch-sequences.md` — the reuse this is supposed to make visible
* engram id:489 (reuse was assumed and was not happening; the meter is what caught it)
* engram id:488 (the bridge's llama-server-shaped usage block, as patched)

## Correction (2026-08-19): `prompt_tokens` is MISLABELLED, not missing

Measured while chasing the canonical-commit defect: the server's `prompt_tokens` does not report the
prompt at all, it reports the tokens actually PREFILLED. A turn whose trace read
`n_prompt=27 n_common=21` was reported to the client as `prompt_tokens=6`, and a turn that reused 29
of 30 tokens was reported as `prompt_tokens=1`.

So the reuse meter is already there, wearing the wrong name — and it is wrong in the direction that
hides the problem: a client watching `prompt_tokens` sees a number that FALLS as reuse improves and
cannot tell that apart from a shorter prompt. Two things to fix together:

* `prompt_tokens` should be the whole prompt (OpenAI semantics), and
* `prompt_tokens_details.cached_tokens` should carry `n_common`, which is the number this field is
  currently standing in for.

Separately, `finish_reason` is `stop` on turns that hit the `max_tokens` cap; it should be `length`.
Confirmed deliberately: a request for exactly 900 tokens generated 900 and still reported `stop`.
This matters more than it looks on a thinking model, where hitting the cap is how an answer comes
back EMPTY (the whole cap spent on reasoning) — and `stop` tells the client that was a normal end.
