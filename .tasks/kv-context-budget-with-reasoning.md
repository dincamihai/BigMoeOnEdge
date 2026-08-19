# Context budget when reasoning is retained, and the n_rs_seq rollback window

column: Todo
created: 2026-08-19

Split off deliberately from `kv-canonical-scratch-sequences.md` (user direction 2026-08-19): the
reuse fix lands first and alone, this is the follow-up.

## Why this is a card and not a paragraph

The canonical/scratch design keeps reasoning OUT of sequence 0, so the ~5-8x context inflation it
would otherwise cause does not happen. That is the whole reason it wins over the simpler "keep the
reasoning in the KV" design. So this card is NOT blocking, and it is not a defect in that design.

What it holds is the material that argued the decision, so the next person does not re-derive it,
plus the two questions that survive.

## The inflation number

From the 2026-08-19 measurement on DSv4-Flash: turn 1 was 13 prompt tokens against 92 generated,
whose parsed content is `1, 2, 3.` — roughly 85% of the growth is reasoning. Any design that
retains reasoning in the served context fills `n_ctx` about 5-8x faster on this model. On a short
`-c` that is turns, not conversations.

## The n_rs_seq finding (worth keeping, do not act on it)

This fork's recurrent memory can roll back partially. `llama-memory-recurrent.cpp` in `seq_rm`:

    // partial rollback via per-token snapshot index (bounded by n_rs_seq)
    if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
        const llama_pos rollback = cell.pos - (p0 - 1);
        if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) { ...; return true; }
        return false;

So the refusal we measured was NOT "a recurrent state cannot be cut". It was a rollback of 93 and
then 158 tokens against a window of zero: `n_rs_seq` defaults to 0 and is set only from
`params.speculative.need_n_rs_seq()`, sized to the speculative draft depth.

Widening it is not a general fix and should not be attempted as one. The snapshots are paid for in
tensor width — `const uint32_t n_rows = mem_size * (1 + n_rs_seq)` — so covering a rollback of N
tokens costs (1+N) times the recurrent state, and `n_gen` is unbounded: a long thinking answer is
thousands of tokens. It is the right mechanism for a speculative draft and the wrong one here.

Also recorded, because it was checked and is load-bearing for the scratch design:
`llama_memory_recurrent::seq_cp` does not copy a state, it makes the destination ALIAS the source's
tail cell. That is safe because `find_slot` is copy-on-write — a cell with more than one `seq_id`
is forked into a fresh cell carrying the original's `src`. Decoding into the copy leaves the source
untouched. If that ever changes, the scratch-sequence design breaks silently.

## What the shipped fix changed about this question (2026-08-19)

`kv-canonical-scratch-sequences.md` landed, so the premise above is now half-obsolete and the card
should be read with that in mind. The canonical sequence grows at the CLEAN rate — the 5-8x
inflation does not happen, and the "drop the old reasoning" idea has nothing to drop.

What replaced it is a narrower and more concrete pressure: the WORKING sequence still holds the
whole turn, prompt plus every reasoning token, because that is what the model is generating into.
So the peak context of a session is no longer "the conversation" but

    len(canonical) + this turn's n_gen

and `n_gen` is where the reasoning lives. Measured on DSv4-Flash at `-c 4096`, a 3-turn toy
conversation reached canonical=52 with n_gen up to 220, so the peak was dominated entirely by the
turn in flight. A long thinking answer on a long conversation is what actually approaches n_ctx now,
and it does so within a SINGLE turn rather than by accumulating across them.

That reframes the open items below: a reset that trims history does nothing for a turn whose own
reasoning is what fills the window. Whatever is designed here has to handle the in-turn case, which
is closer to "what happens when a turn cannot finish" than to a budgeting policy.

## What is actually open

MEASURED 2026-08-19 (see "The numbers" below), which closes item 2 and reshapes item 1.

1. **Long conversations end by being REFUSED, not by breaking.** The window is enforced by an
   admission check before any decoding: `prompt + n_predict > n_ctx` returns HTTP 500 with
   "prompt + n_predict exceeds the session n_ctx (N); open the session with a larger n_ctx",
   in 0.0s, before work starts. No silent truncation, no confident wrong answer, no crash.
   So a reset is a USABILITY feature, not a correctness requirement — which lowers this card's
   priority but does not remove it.
   What still has to be decided is the trigger (a fraction of `n_ctx`) and the drop set. For the
   drop set see the retraction-driven proposal in engram id:516: drop turns that are SUPERSEDED
   rather than turns that are OLD.
2. ~~Nothing measures this yet.~~ DONE — numbers below.

## The numbers (2026-08-19, DSv4-Flash on evo, after the id:515 sentinel fix)

Substantive Q&A, one question per turn, answers fed back verbatim:

    turn   n_gen   canonical   prefilled
     1      159     32 ->  66     32
     2      456     66 -> 365     21
     3     2048     (empty)       24
     4      965        -> 479    384
     5      571    479 -> 683     18
     6     2048     (empty)       22
     7     1479       -> 830    707

* Canonical grows about **120 net tokens per turn** on this workload.
* `n_gen` ran **159-2048**, and **2 of 7** substantive turns spent the WHOLE 2048-token cap on
  reasoning and returned NO CONTENT AT ALL. That is the in-turn pressure, and it is the failure
  users meet first — long before any refusal.
* Peak is `canonical + n_gen`, so turns-per-context is roughly `(n_ctx - max n_gen) / 120`:
  ~17 turns at `-c 4096`, ~250 at evo's usual `-c 32768`. But a SINGLE turn can still exhaust
  the window on its own, which is the real risk and is independent of conversation length.
* The 384- and 707-token prefills on turns 4 and 7 are a discarded turn being dropped from the
  history: a history that is not an extension of canonical rebuilds it from zero. An accidental
  but faithful measurement of what an interior history edit costs.

### How to measure this cheaply, because the obvious way is wrong

The first attempt drove real questions at `-c 4096` and would have taken over an hour, because it
waited on the MODEL when the subject is the WINDOW. Take the model out of the loop instead:

* Shrink `n_ctx` (`-c 1024`) so the wall arrives on turn 2 or 3 rather than turn 15.
* Set `max_tokens` to 4 and grow the prompt with known-size filler, so each turn costs seconds and
  canonical growth is exact arithmetic rather than hostage to how long the model reasons.

Same event, about 50x cheaper. Probe kept at `/tmp/kvwall.py` (throwaway).

Trap for whoever writes the next driver: do NOT substitute placeholder text when the model returns
an empty answer. Feeding back an assistant message the model never produced diverges the history
from canonical and forces a full re-prefill — it looks exactly like an engine bug (engram id:489).

## Related

* `.tasks/kv-canonical-scratch-sequences.md` — the fix this defers to
* `.tasks/kv-prefix-reuse-recurrent-memory.md` — the diagnosis and the raw numbers
* engram id:506 (context size is a ceiling, not a cost: you pay for tokens actually sent)
